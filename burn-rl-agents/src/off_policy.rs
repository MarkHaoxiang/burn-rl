use std::marker::PhantomData;

use burn::tensor::backend::AutodiffBackend;
use burn_rl::environment::Space;
use burn_rl::module::component::Actor;
use burn_rl::{
    data::{memory::Memory, util::Transition},
    environment::Environment,
};
use rand::Rng;

pub trait OffPolicyAgent<B: AutodiffBackend, TBatch>: Actor {
    fn update(&mut self, batch: TBatch);
}

pub struct OffPolicyAlgorithm<B, E, A, M, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: OffPolicyAgent<B, Vec<Transition<E>>, A = E::A, O = E::O>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    R: Rng,
{
    env: E,
    agent: A,
    memory: M,
    rng: R,
    _phantom: PhantomData<B>,
}

impl<B, E, A, M, R> OffPolicyAlgorithm<B, E, A, M, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: OffPolicyAgent<B, Vec<Transition<E>>, A = E::A, O = E::O>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    R: Rng,
{
    pub fn train(mut self) {
        // Early Start
        let mut observation = self.env.reset();
        for _ in 0..1000 {
            let action = <E::A>::sample(&mut self.rng);
            let (next_observation, reward, done) = self.env.step(action.clone());
            self.memory.push(Transition {
                before: observation,
                action,
                after: next_observation.clone(),
                reward,
                done,
            });
            if done {
                observation = self.env.reset();
            } else {
                observation = next_observation;
            }
        }

        // Main Training Loop
        observation = self.env.reset();
        for _ in 0..1000 {
            // Step Environment
            let action = self.agent.a(&observation);
            let (next_observation, reward, done) = self.env.step(action.clone());
            self.memory.push(Transition {
                before: observation,
                action,
                after: next_observation.clone(),
                reward,
                done,
            });
            if done {
                observation = self.env.reset();
            } else {
                observation = next_observation;
            }
            // Update Agent
            self.agent.update(self.memory.sample_random_batch(32));
        }
    }
}
