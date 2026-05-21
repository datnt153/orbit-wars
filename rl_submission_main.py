import baseline_bot
import ppo_strategy

ppo_strategy.set_baseline_module(baseline_bot)


def agent(obs, config=None):
    return ppo_strategy.agent(obs, config=config)


__all__ = ["agent"]
