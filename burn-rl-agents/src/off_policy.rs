use std::marker::PhantomData;

use burn::config::Config;
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
use tqdm::tqdm;

pub trait OffPolicyAgent<B: AutodiffBackend, TBatch>: Actor {
    fn update(self, batch: TBatch) -> Self;
}

#[derive(Config)]
pub struct OffPolicyAlgorithmConfig {
    #[config(default = 10_000)]
    early_start_steps: u64,
    training_steps: u64,
    batch_size: usize,
}

pub struct OffPolicyAlgorithm<B, E, A, M, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: OffPolicyAgent<B, Vec<Transition<E>>, A = E::A, O = E::O>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    R: Rng,
{
    cfg: OffPolicyAlgorithmConfig,
    env: E,
    agent: A,
    memory: M,
    rng: R,
    _phantom: PhantomData<B>,
}

impl OffPolicyAlgorithmConfig {
    pub fn init<A, B, E, M, R>(
        &self,
        env: E,
        agent: A,
        memory: M,
        rng: R,
    ) -> OffPolicyAlgorithm<B, E, A, M, R>
    where
        E: Environment,
        B: AutodiffBackend,
        A: OffPolicyAgent<B, Vec<Transition<E>>, A = E::A, O = E::O>,
        M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
        R: Rng,
    {
        OffPolicyAlgorithm {
            cfg: self.clone(),
            env,
            agent,
            memory,
            rng,
            _phantom: Default::default(),
        }
    }
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
        let transitions =
            collect_multiple(&mut self.env, None, &mut policy, self.cfg.early_start_steps);
        let _ = transitions.into_iter().map(|x| self.memory.push(x));

        // Main Training Loop
        let mut evaluation_statistics = Vec::new();
        let mut observation = Some(self.env.reset());
        for step in tqdm(0..self.cfg.training_steps) {
            // Step Environment
            let transition = collect_single(&mut self.env, observation, &mut |o| {
                epsilon_greedy(0.1, self.agent.a(o), &mut self.rng)
            });
            observation = Some(transition.after.clone());
            self.memory.push(transition);

            // Update Agent
            self.agent = self
                .agent
                .update(self.memory.sample_random_batch(self.cfg.batch_size));

            // Evaluate
            if step % 1_000 == 0 {
                let episode_reward = evaluate_episode(&mut self.env, &mut |o| self.agent.a(o));
                println!("Episode reward: {}", episode_reward);
                evaluation_statistics.push(episode_reward);
            }
        }

        println!("{:?}", evaluation_statistics);
    }
}
