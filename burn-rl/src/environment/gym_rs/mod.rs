use gym_rs::core::Env;

use super::Environment;

pub struct GymEnvironment<T: Env> {
    env: T,
}

impl<T: Env> Environment for GymEnvironment<T> {
    type Action = T::Action;

    type Config = T;

    type Observation = T::Observation;

    fn new(env: Self::Config) -> Self {
        GymEnvironment { env }
    }

    fn reset(&mut self) -> Self::Observation {
        let (obs, _) = self.env.reset(None, false, None);
        obs
    }

    fn step(&mut self, action: Self::Action) -> (Self::Observation, super::Reward, super::Done) {
        let action_reward = self.env.step(action);
        (
            action_reward.observation,
            *action_reward.reward.as_ref(),
            action_reward.done,
        )
    }
}
