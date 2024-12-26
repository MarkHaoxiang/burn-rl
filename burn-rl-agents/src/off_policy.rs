use std::marker::PhantomData;

use burn::tensor::backend::AutodiffBackend;
use burn_rl::data::util::{collect_multiple, collect_single};
use burn_rl::environment::Space;
use burn_rl::logging::evaluate_episode;
use burn_rl::module::component::Actor;
use burn_rl::module::exploration::epsilon_greedy;
use burn_rl::{
    data::{memory::Memory, util::Transition},
    environment::Environment,
};
use rand::Rng;

pub trait OffPolicyAgent<B: AutodiffBackend, TBatch>: Actor {
    fn update(self, batch: TBatch) -> Self;
}

pub struct OffPolicyAlgorithm<B, E, A, M, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: OffPolicyAgent<B, Vec<Transition<E>>, A = E::A, O = E::O>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    R: Rng,
{
    pub env: E,
    pub agent: A,
    pub memory: M,
    pub rng: R,
    pub _phantom: PhantomData<B>,
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
        let mut policy = |_: &E::O| <E::A>::sample(&mut self.rng);
        let transitions = collect_multiple(&mut self.env, None, &mut policy, 2000);
        let _ = transitions.into_iter().map(|x| self.memory.push(x));

        // Main Training Loop
        let mut observation = Some(self.env.reset());
        for step in 0..10000 {
            // Step Environment
            let transition = collect_single(&mut self.env, observation, &mut |o| {
                epsilon_greedy(0.1, self.agent.a(o), &mut self.rng)
            });
            observation = Some(transition.after.clone());
            self.memory.push(transition);

            // Update Agent
            self.agent = self.agent.update(self.memory.sample_random_batch(64));

            // Evaluate
            if step % 100 == 0 {
                println!(
                    "Episode reward: {}",
                    evaluate_episode(&mut self.env, &mut |o| self.agent.a(o))
                );
            }
        }
    }
}
