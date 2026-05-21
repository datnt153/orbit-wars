// R1 (DONE, parity verified) + R2 — Rust port of src/fast_sim.py with
// PyO3 bindings, vectorized batched-step over many envs via rayon.
//
// step() splits into:
//   extract_joint_actions(...) → Rust Vec<Vec<(i64,f64,i64)>>  (with GIL)
//   step_pure(actions, num_agents)                              (no Python)
// → EnvPool.step_batch releases the GIL and runs step_pure across all
// envs with rayon — true parallel throughput across cores.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use ndarray::{Array3, Array4};
use numpy::IntoPyArray;

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = 50.0;
const SUN_RADIUS: f64 = 10.0;
const ROT_LIMIT: f64 = 50.0;

// Feature encoder constants — must mirror src/policy_encode.py
pub const MAXP_R: usize = 48;
pub const F_R: usize = 15;
pub const G_R: usize = 8;
const COMET_STEPS: [i64; 5] = [50, 150, 250, 350, 450];

#[inline]
fn distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

#[inline]
fn point_to_segment_distance(px: f64, py: f64, vx: f64, vy: f64, wx: f64, wy: f64) -> f64 {
    let l2 = (vx - wx).powi(2) + (vy - wy).powi(2);
    if l2 == 0.0 { return distance(px, py, vx, vy); }
    let mut t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
    if t < 0.0 { t = 0.0; }
    if t > 1.0 { t = 1.0; }
    let projx = vx + t * (wx - vx);
    let projy = vy + t * (wy - vy);
    distance(px, py, projx, projy)
}

#[derive(Clone, Debug)]
struct Planet { id: i64, owner: i64, x: f64, y: f64, r: f64, ships: i64, prod: i64 }

#[derive(Clone, Debug)]
struct Fleet { id: i64, owner: i64, x: f64, y: f64, angle: f64, from_id: i64, ships: i64 }

#[derive(Clone, Debug)]
struct CometGroup { planet_ids: Vec<i64>, paths: Vec<Vec<(f64, f64)>>, path_index: i64 }

#[pyclass]
#[derive(Clone)]
pub struct State {
    planets: Vec<Planet>,
    initial_planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    comets: Vec<CometGroup>,
    comet_planet_ids: Vec<i64>,
    next_fleet_id: i64,
    angular_velocity: f64,
    step_no: i64,
    ship_speed: f64,
    episode_steps: i64,
    done: bool,
    scores: Option<Vec<i64>>,
    rewards: Option<Vec<i32>>,
}

fn _g<'py>(obs: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(d) = obs.downcast::<PyDict>() {
        if let Some(v) = d.get_item(key)? { return Ok(v); }
    }
    obs.getattr(key)
}

fn _g_or<'py>(obs: &Bound<'py, PyAny>, key: &str, default: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    _g(obs, key).unwrap_or(default)
}

fn read_planet_list(obj: &Bound<PyAny>) -> PyResult<Vec<Planet>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let it = item?;
        out.push(Planet {
            id: it.get_item(0)?.extract()?,
            owner: it.get_item(1)?.extract()?,
            x: it.get_item(2)?.extract()?,
            y: it.get_item(3)?.extract()?,
            r: it.get_item(4)?.extract()?,
            ships: it.get_item(5)?.extract()?,
            prod: it.get_item(6)?.extract()?,
        });
    }
    Ok(out)
}

fn read_fleet_list(obj: &Bound<PyAny>) -> PyResult<Vec<Fleet>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let it = item?;
        out.push(Fleet {
            id: it.get_item(0)?.extract()?,
            owner: it.get_item(1)?.extract()?,
            x: it.get_item(2)?.extract()?,
            y: it.get_item(3)?.extract()?,
            angle: it.get_item(4)?.extract()?,
            from_id: it.get_item(5)?.extract()?,
            ships: it.get_item(6)?.extract()?,
        });
    }
    Ok(out)
}

fn read_comet_list(obj: &Bound<PyAny>) -> PyResult<Vec<CometGroup>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let it = item?;
        let d = it.downcast::<PyDict>()?;
        let pids: Vec<i64> = d.get_item("planet_ids")?.unwrap().extract()?;
        let paths_obj = d.get_item("paths")?.unwrap();
        let mut paths = Vec::with_capacity(pids.len());
        for p in paths_obj.iter()? {
            let pp = p?;
            let pts: Vec<(f64, f64)> = pp.extract()?;
            paths.push(pts);
        }
        let path_index: i64 = d.get_item("path_index")?.unwrap().extract()?;
        out.push(CometGroup { planet_ids: pids, paths, path_index });
    }
    Ok(out)
}

/// Extract joint actions [agent][move] from a Python list-of-lists into a
/// pure-Rust Vec so step_pure can run without the GIL.
fn extract_joint_actions(
    joint_actions: &Bound<'_, PyList>, num_agents: usize,
) -> PyResult<Vec<Vec<(i64, f64, i64)>>> {
    let mut out = Vec::with_capacity(num_agents);
    for pid in 0..num_agents {
        let action = match joint_actions.get_item(pid) {
            Ok(a) => a, Err(_) => { out.push(Vec::new()); continue; }
        };
        if action.is_none() { out.push(Vec::new()); continue; }
        let moves = match action.downcast::<PyList>() {
            Ok(l) => l.clone(), Err(_) => { out.push(Vec::new()); continue; }
        };
        let mut mv_vec = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            let ml = match mv.downcast::<PyList>() {
                Ok(l) => l.clone(), Err(_) => continue,
            };
            if ml.len() != 3 { continue; }
            let from_id: i64 = match ml.get_item(0).and_then(|v| v.extract()) { Ok(v) => v, Err(_) => continue };
            let angle: f64 = match ml.get_item(1).and_then(|v| v.extract()) { Ok(v) => v, Err(_) => continue };
            let ships: i64 = match ml.get_item(2).and_then(|v| v.extract()) { Ok(v) => v, Err(_) => continue };
            mv_vec.push((from_id, angle, ships));
        }
        out.push(mv_vec);
    }
    Ok(out)
}

impl State {
    /// Rust port of src/policy_encode.encode_state for one (state, player).
    /// Fills slices in caller-allocated buffers — zero allocations for
    /// max speed inside the per-env loop (called once per player per env).
    /// Must produce float32 outputs matching the Python encoder within
    /// f32 precision (parity-tested in observe_parity.py).
    fn observe_one(&self, player: i64,
                    pf: &mut [f32], pmask: &mut [f32],
                    omask: &mut [f32], gf: &mut [f32], pids: &mut [i64]) {
        // Reset omask (pf/pmask/gf/pids reset by caller via zeros/-1 fill).
        for v in omask.iter_mut() { *v = 0.0; }
        let n = self.planets.len().min(MAXP_R);

        // Pressure per planet (player-relative).
        let mut press_mine = [0.0_f32; MAXP_R];
        let mut press_enemy = [0.0_f32; MAXP_R];
        for f in &self.fleets {
            let fa = f.angle;
            let fx = f.x; let fy = f.y;
            let mut best_j: i64 = -1;
            let mut best_e: f64 = 1e9;
            for (j, p) in self.planets.iter().take(n).enumerate() {
                let ang = (p.y - fy).atan2(p.x - fx);
                let d = (fa - ang).rem_euclid(2.0 * std::f64::consts::PI);
                let err = d.min(2.0 * std::f64::consts::PI - d);
                if err < best_e {
                    best_e = err;
                    best_j = j as i64;
                }
            }
            if best_j >= 0 && best_e < 0.5 {
                let j = best_j as usize;
                if f.owner == player {
                    press_mine[j] += f.ships as f32;
                } else if f.owner >= 0 {
                    press_enemy[j] += f.ships as f32;
                }
            }
        }

        let is_4p = self.planets.iter().any(|p| p.owner >= 2);
        let comet_ids: HashSet<i64> = self.comet_planet_ids.iter().copied().collect();
        let mut my_tot = 0.0_f32;
        let mut opp_tot = 0.0_f32;
        let mut neu_tot = 0.0_f32;
        let mut my_np = 0_usize;

        for (i, p) in self.planets.iter().take(n).enumerate() {
            let own = p.owner;
            let ships = p.ships as f32;
            if own == player {
                my_tot += ships + (p.prod * 8) as f32;
                my_np += 1;
            } else if own == -1 {
                neu_tot += ships;
            } else if own >= 0 {
                opp_tot += ships + (p.prod * 8) as f32;
            }
            let dx = p.x - CENTER;
            let dy = p.y - CENTER;
            let dsun = (dx*dx + dy*dy).sqrt();
            let is_orb = if dsun + p.r < ROT_LIMIT { 1.0_f32 } else { 0.0_f32 };
            let is_mine = if own == player { 1.0_f32 } else { 0.0_f32 };
            let is_enemy = if own >= 0 && own != player { 1.0_f32 } else { 0.0_f32 };
            let is_neutral = if own == -1 { 1.0_f32 } else { 0.0_f32 };
            let is_comet = if comet_ids.contains(&p.id) { 1.0_f32 } else { 0.0_f32 };
            let base = i * F_R;
            pf[base + 0]  = is_mine;
            pf[base + 1]  = is_enemy;
            pf[base + 2]  = is_neutral;
            pf[base + 3]  = (ships as f64).ln_1p() as f32 / 8.0;
            pf[base + 4]  = p.prod as f32 / 5.0;
            pf[base + 5]  = p.r as f32 / 4.0;
            pf[base + 6]  = p.x as f32 / 100.0;
            pf[base + 7]  = p.y as f32 / 100.0;
            pf[base + 8]  = (dsun / 70.0) as f32;
            pf[base + 9]  = is_comet;
            pf[base + 10] = is_orb;
            pf[base + 11] = (press_mine[i] as f64).ln_1p() as f32 / 8.0;
            pf[base + 12] = (press_enemy[i] as f64).ln_1p() as f32 / 8.0;
            pf[base + 13] = is_mine;
            pf[base + 14] = 1.0;
            pmask[i] = 1.0;
            pids[i] = p.id;
            if own == player { omask[i] = 1.0; }
        }
        let step = self.step_no;
        let nxt = COMET_STEPS.iter().copied().find(|&c| c > step).unwrap_or(500);
        let sc = 200.0_f32;
        gf[0] = step as f32 / 500.0;
        gf[1] = if is_4p { 1.0 } else { 0.0 };
        gf[2] = my_tot / sc;
        gf[3] = opp_tot / sc;
        gf[4] = neu_tot / sc;
        gf[5] = my_np as f32 / 20.0;
        gf[6] = (nxt - step) as f32 / 100.0;
        gf[7] = my_np as f32 / (n.max(1)) as f32;
    }

    /// Pure Rust step — no Python, no GIL needed. Used directly by rayon
    /// inside EnvPool.step_batch.
    fn step_pure(&mut self, actions: &[Vec<(i64, f64, i64)>], num_agents: usize) {
        let av = self.angular_velocity;
        let step_no = self.step_no;

        // --- 0a. Remove expired comets ---
        let mut expired: Vec<i64> = Vec::new();
        for grp in &self.comets {
            let idx = grp.path_index as usize;
            for (i, pid) in grp.planet_ids.iter().enumerate() {
                if idx >= grp.paths[i].len() { expired.push(*pid); }
            }
        }
        if !expired.is_empty() {
            let eset: std::collections::HashSet<i64> = expired.iter().copied().collect();
            self.planets.retain(|p| !eset.contains(&p.id));
            self.initial_planets.retain(|p| !eset.contains(&p.id));
            self.comet_planet_ids.retain(|pid| !eset.contains(pid));
            for g in self.comets.iter_mut() {
                g.planet_ids.retain(|pid| !eset.contains(pid));
            }
            self.comets.retain(|g| !g.planet_ids.is_empty());
        }

        // --- 1. Fleet launch ---
        let pmap: HashMap<i64, usize> = self.planets.iter().enumerate().map(|(i, p)| (p.id, i)).collect();
        let max_speed = self.ship_speed;
        for pid in 0..num_agents.min(actions.len()) {
            for &(from_id, angle, ships) in &actions[pid] {
                if ships <= 0 { continue; }
                if let Some(&pi) = pmap.get(&from_id) {
                    let fp = &mut self.planets[pi];
                    if fp.owner == pid as i64 && fp.ships >= ships {
                        fp.ships -= ships;
                        let sx = fp.x + angle.cos() * (fp.r + 0.1);
                        let sy = fp.y + angle.sin() * (fp.r + 0.1);
                        let fid = self.next_fleet_id;
                        self.fleets.push(Fleet {
                            id: fid, owner: pid as i64, x: sx, y: sy,
                            angle, from_id, ships,
                        });
                        self.next_fleet_id += 1;
                    }
                }
            }
        }

        // --- 2. Production ---
        for p in self.planets.iter_mut() {
            if p.owner != -1 { p.ships += p.prod; }
        }

        // --- 3. Fleet movement + continuous collision ---
        let mut to_remove_idx: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut combat: HashMap<i64, Vec<Fleet>> = HashMap::new();
        for p in &self.planets { combat.insert(p.id, Vec::new()); }
        for (fi, fleet) in self.fleets.iter_mut().enumerate() {
            let angle = fleet.angle;
            let ships = fleet.ships as f64;
            let mut speed = 1.0 + (max_speed - 1.0) * (ships.ln() / 1000f64.ln()).powf(1.5);
            if speed > max_speed { speed = max_speed; }
            let ox = fleet.x; let oy = fleet.y;
            fleet.x += angle.cos() * speed;
            fleet.y += angle.sin() * speed;
            let nx = fleet.x; let ny = fleet.y;
            if !(nx >= 0.0 && nx <= BOARD_SIZE && ny >= 0.0 && ny <= BOARD_SIZE) {
                to_remove_idx.insert(fi); continue;
            }
            if point_to_segment_distance(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS {
                to_remove_idx.insert(fi); continue;
            }
            for p in &self.planets {
                if point_to_segment_distance(p.x, p.y, ox, oy, nx, ny) < p.r {
                    combat.get_mut(&p.id).unwrap().push(fleet.clone());
                    to_remove_idx.insert(fi);
                    break;
                }
            }
        }

        // --- 4. Planet rotation + sweep ---
        let comet_pids: std::collections::HashSet<i64> = self.comet_planet_ids.iter().copied().collect();
        let init_by_id: HashMap<i64, &Planet> = self.initial_planets.iter().map(|p| (p.id, p)).collect();

        let ip_lookup: Vec<(f64, f64, f64)> = self.planets.iter().map(|p| {
            if comet_pids.contains(&p.id) { return (0.0, 0.0, 0.0); }
            if let Some(ip) = init_by_id.get(&p.id) {
                let dx = ip.x - CENTER; let dy = ip.y - CENTER;
                let rr = (dx * dx + dy * dy).sqrt();
                let ia = dy.atan2(dx);
                (rr, ia, 1.0)
            } else { (0.0, 0.0, 0.0) }
        }).collect();

        for (pi, planet) in self.planets.iter_mut().enumerate() {
            if comet_pids.contains(&planet.id) { continue; }
            let (rr, ia, valid) = ip_lookup[pi];
            if valid == 0.0 { continue; }
            let ox = planet.x; let oy = planet.y;
            if rr + planet.r < ROT_LIMIT {
                let ca = ia + av * step_no as f64;
                planet.x = CENTER + rr * ca.cos();
                planet.y = CENTER + rr * ca.sin();
            }
            if ox != planet.x || oy != planet.y {
                for (fi, fleet) in self.fleets.iter().enumerate() {
                    if to_remove_idx.contains(&fi) { continue; }
                    if point_to_segment_distance(fleet.x, fleet.y, ox, oy, planet.x, planet.y) < planet.r {
                        combat.get_mut(&planet.id).unwrap().push(fleet.clone());
                        to_remove_idx.insert(fi);
                    }
                }
            }
        }

        // Comet movement
        let mut expired2: Vec<i64> = Vec::new();
        let mut comet_planet_updates: Vec<(i64, f64, f64, f64, f64)> = Vec::new();
        for grp in self.comets.iter_mut() {
            grp.path_index += 1;
            let idx = grp.path_index as usize;
            for (i, pid) in grp.planet_ids.iter().enumerate() {
                let pi = match self.planets.iter().position(|p| p.id == *pid) { Some(v) => v, None => continue };
                let ppath = &grp.paths[i];
                if idx >= ppath.len() {
                    expired2.push(*pid);
                } else {
                    let ox = self.planets[pi].x; let oy = self.planets[pi].y;
                    let nx = ppath[idx].0; let ny = ppath[idx].1;
                    self.planets[pi].x = nx;
                    self.planets[pi].y = ny;
                    if ox >= 0.0 {
                        comet_planet_updates.push((*pid, ox, oy, nx, ny));
                    }
                }
            }
        }
        for (pid, ox, oy, nx, ny) in &comet_planet_updates {
            let pi = match self.planets.iter().position(|p| p.id == *pid) { Some(v) => v, None => continue };
            let radius = self.planets[pi].r;
            for (fi, fleet) in self.fleets.iter().enumerate() {
                if to_remove_idx.contains(&fi) { continue; }
                if point_to_segment_distance(fleet.x, fleet.y, *ox, *oy, *nx, *ny) < radius {
                    combat.get_mut(pid).unwrap().push(fleet.clone());
                    to_remove_idx.insert(fi);
                }
            }
        }

        if !expired2.is_empty() {
            let eset: std::collections::HashSet<i64> = expired2.iter().copied().collect();
            self.planets.retain(|p| !eset.contains(&p.id));
            self.initial_planets.retain(|p| !eset.contains(&p.id));
            self.comet_planet_ids.retain(|pid| !eset.contains(pid));
            for g in self.comets.iter_mut() {
                g.planet_ids.retain(|pid| !eset.contains(pid));
            }
            self.comets.retain(|g| !g.planet_ids.is_empty());
            for k in eset.iter() { combat.remove(k); }
        }

        // Filter fleets
        let mut new_fleets = Vec::with_capacity(self.fleets.len());
        for (i, f) in self.fleets.drain(..).enumerate() {
            if !to_remove_idx.contains(&i) { new_fleets.push(f); }
        }
        self.fleets = new_fleets;

        // --- 5. Combat resolution ---
        let pmap2: HashMap<i64, usize> = self.planets.iter().enumerate().map(|(i, p)| (p.id, i)).collect();
        for (pid, pf) in combat.iter() {
            let planet_idx = match pmap2.get(pid) { Some(v) => *v, None => continue };
            if pf.is_empty() { continue; }
            let mut player_ships: HashMap<i64, i64> = HashMap::new();
            for fl in pf { *player_ships.entry(fl.owner).or_insert(0) += fl.ships; }
            if player_ships.is_empty() { continue; }
            let mut sp: Vec<(i64, i64)> = player_ships.into_iter().collect();
            sp.sort_by(|a, b| b.1.cmp(&a.1));
            let (top_player, top_ships) = sp[0];
            let (surv, surv_owner) = if sp.len() > 1 {
                let second = sp[1].1;
                let s = top_ships - second;
                if sp[0].1 == sp[1].1 { (0, -1) }
                else { (s, if s > 0 { top_player } else { -1 }) }
            } else { (top_ships, top_player) };
            if surv > 0 {
                let planet = &mut self.planets[planet_idx];
                if planet.owner == surv_owner { planet.ships += surv; }
                else {
                    planet.ships -= surv;
                    if planet.ships < 0 {
                        planet.owner = surv_owner;
                        planet.ships = -planet.ships;
                    }
                }
            }
        }

        // --- 6. Termination + scoring ---
        self.step_no += 1;
        let mut terminated = false;
        if self.step_no >= self.episode_steps - 2 { terminated = true; }
        let mut alive: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for p in &self.planets { if p.owner != -1 { alive.insert(p.owner); } }
        for f in &self.fleets { alive.insert(f.owner); }
        if alive.len() <= 1 { terminated = true; }
        self.done = terminated;
        if terminated {
            let mut scores = vec![0i64; num_agents];
            for p in &self.planets {
                if p.owner != -1 && (p.owner as usize) < num_agents { scores[p.owner as usize] += p.ships; }
            }
            for f in &self.fleets {
                if (f.owner as usize) < num_agents { scores[f.owner as usize] += f.ships; }
            }
            let mx = *scores.iter().max().unwrap_or(&0);
            let rewards: Vec<i32> = scores.iter().map(|&s| if s == mx && mx > 0 { 1 } else { -1 }).collect();
            self.scores = Some(scores);
            self.rewards = Some(rewards);
        }
    }
}

#[pymethods]
impl State {
    #[new]
    #[pyo3(signature = (obs, ship_speed=6.0, episode_steps=500))]
    fn new(obs: &Bound<'_, PyAny>, ship_speed: f64, episode_steps: i64) -> PyResult<Self> {
        let py = obs.py();
        let empty_list = PyList::empty_bound(py);
        let planets = read_planet_list(&_g_or(obs, "planets", empty_list.clone().into_any()))?;
        let fleets = read_fleet_list(&_g_or(obs, "fleets", empty_list.clone().into_any()))?;
        let initial_planets = read_planet_list(&_g_or(obs, "initial_planets", empty_list.clone().into_any()))?;
        let comets = read_comet_list(&_g_or(obs, "comets", empty_list.clone().into_any()))?;
        let comet_planet_ids: Vec<i64> = _g_or(obs, "comet_planet_ids", empty_list.clone().into_any())
            .extract().unwrap_or_default();
        let next_fleet_id: i64 = _g(obs, "next_fleet_id").and_then(|v| v.extract()).unwrap_or(0);
        let angular_velocity: f64 = _g(obs, "angular_velocity").and_then(|v| v.extract()).unwrap_or(0.0);
        let step_no: i64 = _g(obs, "step").and_then(|v| v.extract()).unwrap_or(0);
        Ok(State {
            planets, initial_planets, fleets, comets, comet_planet_ids,
            next_fleet_id, angular_velocity, step_no, ship_speed, episode_steps,
            done: false, scores: None, rewards: None,
        })
    }

    fn clone_state(&self) -> Self { self.clone() }

    #[getter] fn step_no(&self) -> i64 { self.step_no }
    #[getter] fn done(&self) -> bool { self.done }
    #[getter] fn next_fleet_id(&self) -> i64 { self.next_fleet_id }

    fn step(&mut self, joint_actions: &Bound<'_, PyList>, num_agents: usize) -> PyResult<()> {
        let actions = extract_joint_actions(joint_actions, num_agents)?;
        self.step_pure(&actions, num_agents);
        Ok(())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let pl: Vec<Vec<f64>> = self.planets.iter().map(|p| vec![
            p.id as f64, p.owner as f64, p.x, p.y, p.r, p.ships as f64, p.prod as f64
        ]).collect();
        d.set_item("planets", pl)?;
        let fl: Vec<Vec<f64>> = self.fleets.iter().map(|f| vec![
            f.id as f64, f.owner as f64, f.x, f.y, f.angle, f.from_id as f64, f.ships as f64
        ]).collect();
        d.set_item("fleets", fl)?;
        let ip: Vec<Vec<f64>> = self.initial_planets.iter().map(|p| vec![
            p.id as f64, p.owner as f64, p.x, p.y, p.r, p.ships as f64, p.prod as f64
        ]).collect();
        d.set_item("initial_planets", ip)?;
        let cs = PyList::empty_bound(py);
        for g in &self.comets {
            let gd = PyDict::new_bound(py);
            gd.set_item("planet_ids", &g.planet_ids)?;
            let paths = PyList::empty_bound(py);
            for p in &g.paths {
                let p_pts: Vec<(f64, f64)> = p.clone();
                paths.append(p_pts)?;
            }
            gd.set_item("paths", paths)?;
            gd.set_item("path_index", g.path_index)?;
            cs.append(gd)?;
        }
        d.set_item("comets", cs)?;
        d.set_item("comet_planet_ids", &self.comet_planet_ids)?;
        d.set_item("next_fleet_id", self.next_fleet_id)?;
        d.set_item("angular_velocity", self.angular_velocity)?;
        d.set_item("step", self.step_no)?;
        d.set_item("ship_speed", self.ship_speed)?;
        d.set_item("episode_steps", self.episode_steps)?;
        d.set_item("_done", self.done)?;
        if let Some(ref s) = self.scores { d.set_item("_scores", s)?; }
        if let Some(ref r) = self.rewards { d.set_item("_rewards", r)?; }
        Ok(d)
    }
}

// ---------------- R2: EnvPool — N envs, parallel step via rayon -----------

#[pyclass]
pub struct EnvPool { states: Vec<State> }

#[pymethods]
impl EnvPool {
    #[new]
    fn new(template: &State, n: usize) -> Self {
        EnvPool { states: vec![template.clone(); n] }
    }

    #[getter] fn n(&self) -> usize { self.states.len() }

    /// joint_actions_batch: list[n_envs] of list[num_agents] of list[moves]
    fn step_batch(
        &mut self, py: Python, joint_actions_batch: &Bound<'_, PyList>, num_agents: usize,
    ) -> PyResult<()> {
        let n = self.states.len();
        if joint_actions_batch.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("actions batch len {} != n_envs {}", joint_actions_batch.len(), n)));
        }
        // Phase 1: extract all actions to Rust (with GIL).
        let mut acts_rust: Vec<Vec<Vec<(i64, f64, i64)>>> = Vec::with_capacity(n);
        for i in 0..n {
            let ai = joint_actions_batch.get_item(i)?;
            let ali = ai.downcast::<PyList>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("env {}: {}", i, e))
            })?;
            acts_rust.push(extract_joint_actions(ali, num_agents)?);
        }
        // Phase 2: release GIL, rayon parallel step.
        py.allow_threads(|| {
            self.states.par_iter_mut().zip(acts_rust.par_iter()).for_each(|(s, a)| {
                s.step_pure(a, num_agents);
            });
        });
        Ok(())
    }

    /// Per-env done flag.
    fn done_mask(&self) -> Vec<bool> {
        self.states.iter().map(|s| s.done).collect()
    }

    /// Per-env rewards (empty until done).
    fn rewards(&self) -> Vec<Vec<i32>> {
        self.states.iter().map(|s| s.rewards.clone().unwrap_or_default()).collect()
    }

    /// Per-env step_no.
    fn step_nos(&self) -> Vec<i64> {
        self.states.iter().map(|s| s.step_no).collect()
    }

    /// Replace env i with a fresh copy of `template`.
    fn reset_one(&mut self, env_idx: usize, template: &State) -> PyResult<()> {
        if env_idx >= self.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_idx out of range"));
        }
        self.states[env_idx] = template.clone();
        Ok(())
    }

    /// Reset every done env to `template`.
    fn reset_done(&mut self, template: &State) -> usize {
        let mut count = 0;
        for s in self.states.iter_mut() {
            if s.done { *s = template.clone(); count += 1; }
        }
        count
    }

    /// Get the i-th state (debug/inspection — clones).
    fn get_state(&self, i: usize) -> PyResult<State> {
        self.states.get(i).cloned().ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("out of range"))
    }

    /// Batched feature encoder — same semantics as
    /// src/policy_encode.encode_state but in Rust, rayon-parallel.
    /// Returns numpy arrays directly so Python skips the encode_batch
    /// loop entirely (the major bottleneck of CPU-bound rollout).
    ///
    /// Returns (pf, pmask, omask, gf, pids):
    ///   pf [n_envs, num_agents, MAXP, F]
    ///   pmask, omask [n_envs, num_agents, MAXP]
    ///   gf [n_envs, num_agents, G]
    ///   pids list[n_envs] of list[MAXP]
    fn observe_batch<'py>(
        &self, py: Python<'py>, num_agents: usize,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray4<f32>>,
        Bound<'py, numpy::PyArray3<f32>>,
        Bound<'py, numpy::PyArray3<f32>>,
        Bound<'py, numpy::PyArray3<f32>>,
        Vec<Vec<i64>>,
    )> {
        let n_envs = self.states.len();
        let mut pf = Array4::<f32>::zeros((n_envs, num_agents, MAXP_R, F_R));
        let mut pm = Array3::<f32>::zeros((n_envs, num_agents, MAXP_R));
        let mut om = Array3::<f32>::zeros((n_envs, num_agents, MAXP_R));
        let mut gf = Array3::<f32>::zeros((n_envs, num_agents, G_R));
        let mut pids = vec![vec![-1_i64; MAXP_R]; n_envs];

        let pf_data = pf.as_slice_mut().unwrap();
        let pm_data = pm.as_slice_mut().unwrap();
        let om_data = om.as_slice_mut().unwrap();
        let gf_data = gf.as_slice_mut().unwrap();

        let stride_pf  = num_agents * MAXP_R * F_R;
        let stride_pm  = num_agents * MAXP_R;
        let stride_gf  = num_agents * G_R;
        let stride_pp  = MAXP_R * F_R;
        let stride_pmp = MAXP_R;
        let stride_gp  = G_R;

        py.allow_threads(|| {
            pf_data.par_chunks_mut(stride_pf)
                .zip(pm_data.par_chunks_mut(stride_pm))
                .zip(om_data.par_chunks_mut(stride_pm))
                .zip(gf_data.par_chunks_mut(stride_gf))
                .zip(pids.par_iter_mut())
                .zip(self.states.par_iter())
                .for_each(|(((((pf_e, pm_e), om_e), gf_e), pids_e), st)| {
                    for p in 0..num_agents {
                        let pf_p = &mut pf_e[p*stride_pp .. (p+1)*stride_pp];
                        let pm_p = &mut pm_e[p*stride_pmp .. (p+1)*stride_pmp];
                        let om_p = &mut om_e[p*stride_pmp .. (p+1)*stride_pmp];
                        let gf_p = &mut gf_e[p*stride_gp  .. (p+1)*stride_gp];
                        // pids same across players → write into pids_e (slot
                        // -1's get overwritten identically each iteration).
                        st.observe_one(p as i64, pf_p, pm_p, om_p, gf_p, pids_e.as_mut_slice());
                    }
                });
        });

        Ok((
            pf.into_pyarray_bound(py),
            pm.into_pyarray_bound(py),
            om.into_pyarray_bound(py),
            gf.into_pyarray_bound(py),
            pids,
        ))
    }

    /// Build joint actions from per-planet (launch, target, ship_frac)
    /// samples and step all envs in parallel. Eliminates the Python
    /// action-construction loop — the last serial bottleneck.
    /// Shapes: launch [n_envs, num_agents, MAXP] f32 (0/1 floats),
    ///         target [n_envs, num_agents, MAXP] i64 (planet ROW index),
    ///         ship_frac [n_envs, num_agents, MAXP] f32 (0..1, all-in if 1.0).
    fn step_from_samples<'py>(
        &mut self, py: Python<'py>,
        launch: numpy::PyReadonlyArray3<'py, f32>,
        target: numpy::PyReadonlyArray3<'py, i64>,
        ship_frac: numpy::PyReadonlyArray3<'py, f32>,
        num_agents: usize,
    ) -> PyResult<()> {
        let n_envs = self.states.len();
        let l = launch.as_array();
        let tg = target.as_array();
        let sf = ship_frac.as_array();
        if l.shape() != [n_envs, num_agents, MAXP_R] ||
           tg.shape() != [n_envs, num_agents, MAXP_R] ||
           sf.shape() != [n_envs, num_agents, MAXP_R] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expected shape [{},{},{}]", n_envs, num_agents, MAXP_R)));
        }
        // Snapshot ndarrays into owned Vecs so closure is Send.
        let l_owned: Vec<f32> = l.iter().copied().collect();
        let tg_owned: Vec<i64> = tg.iter().copied().collect();
        let sf_owned: Vec<f32> = sf.iter().copied().collect();
        py.allow_threads(|| {
            self.states.par_iter_mut().enumerate().for_each(|(e, st)| {
                // Build joint_actions for this env from samples.
                // Planet list snapshot (id, x, y, ships) to derive bearings.
                let pids: Vec<(i64, f64, f64, i64)> = st.planets.iter()
                    .map(|p| (p.id, p.x, p.y, p.ships))
                    .collect();
                let mut acts: Vec<Vec<(i64, f64, i64)>> = vec![Vec::new(); num_agents];
                let env_base = e * num_agents * MAXP_R;
                for p in 0..num_agents {
                    let p_base = env_base + p * MAXP_R;
                    for r in 0..MAXP_R {
                        if r >= pids.len() { break; }
                        if l_owned[p_base + r] < 0.5 { continue; }
                        let t = tg_owned[p_base + r] as usize;
                        if t >= pids.len() { continue; }
                        let (src_id, sx, sy, src_ships) = pids[r];
                        let (tgt_id, tx, ty, _) = pids[t];
                        if src_id < 0 || tgt_id < 0 || src_ships <= 0 { continue; }
                        let frac = sf_owned[p_base + r].clamp(0.0, 1.0) as f64;
                        let n = ((src_ships as f64) * frac).max(1.0) as i64;
                        let angle = (ty - sy).atan2(tx - sx);
                        acts[p].push((src_id, angle, n));
                    }
                }
                st.step_pure(&acts, num_agents);
            });
        });
        Ok(())
    }

    /// Per-env per-player (ship + 8*prod) totals minus avg opponent — used
    /// as a shaped reward signal during RL training (delta over time gives
    /// dense feedback in the otherwise-sparse Orbit Wars reward).
    fn diff_vs_avg_opp(&self, num_agents: usize) -> Vec<Vec<f64>> {
        self.states.iter().map(|s| {
            let mut totals = vec![0.0_f64; num_agents];
            for p in &s.planets {
                if p.owner >= 0 && (p.owner as usize) < num_agents {
                    totals[p.owner as usize] += (p.ships + p.prod * 8) as f64;
                }
            }
            for f in &s.fleets {
                if (f.owner as usize) < num_agents {
                    totals[f.owner as usize] += f.ships as f64;
                }
            }
            let sum: f64 = totals.iter().sum();
            (0..num_agents).map(|p| {
                let others = sum - totals[p];
                let avg_other = others / ((num_agents - 1).max(1) as f64);
                totals[p] - avg_other
            }).collect()
        }).collect()
    }
}

#[pyfunction]
fn py_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 { distance(ax, ay, bx, by) }

#[pyfunction]
fn py_point_to_segment_distance(px: f64, py: f64, vx: f64, vy: f64, wx: f64, wy: f64) -> f64 {
    point_to_segment_distance(px, py, vx, vy, wx, wy)
}

#[pymodule]
fn ow_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_point_to_segment_distance, m)?)?;
    m.add_class::<State>()?;
    m.add_class::<EnvPool>()?;
    Ok(())
}
