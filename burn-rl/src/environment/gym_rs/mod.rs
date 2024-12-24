use gym_rs::{
    core::Env,
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::custom::traits::Sample,
};

use super::{Environment, Space};

pub struct GymEnvironment<T: Env> {
    env: T,
}

impl<T: Env> GymEnvironment<T> {
    pub fn from(env: T) -> Self {
        GymEnvironment { env }
    }
}

// Unless we change the upstream definition of spaces,
// We need separate implementations for each env
#[derive(Clone)]
pub struct CartPoleAction(<CartPoleEnv as Env>::Action);
impl CartPoleAction {
    pub fn from(a: usize) -> Self {
        assert!(a < 2, "{} cartpole action invalid", a);
        CartPoleAction(a)
    }
}

impl Space for CartPoleObservation {
    fn sample<R: rand::Rng>(rng: &mut R) -> Self {
        CartPoleObservation::sample_between(rng, None)
    }
}

impl Space for CartPoleAction {
    fn sample<R: rand::Rng>(rng: &mut R) -> Self {
        CartPoleAction(rng.gen_range(0..2))
    }
}

impl Environment for GymEnvironment<CartPoleEnv> {
    type A = CartPoleAction;

    type O = CartPoleObservation;

    fn reset(&mut self) -> Self::O {
        self.env.reset(None, false, None).0
    }

    fn step(&mut self, action: Self::A) -> (Self::O, super::Reward, super::Done) {
        let action_reward = self.env.step(action.0);
        (
            action_reward.observation,
            *action_reward.reward.as_ref(),
            action_reward.done,
        )
    }
}
