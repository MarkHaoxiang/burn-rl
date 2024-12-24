use std::marker::PhantomData;

use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamConfig, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, ElementComparison},
};
use burn_rl::{
    data::memory::{Memory, RingbufferMemory},
    environment::{
        gym_rs::{CartPoleAction, GymEnvironment},
        Environment, Space,
    },
    module::{
        abc::{Actor, Critic, Value},
        nn::multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
    },
    objective::temporal_difference::temporal_difference,
};
use gym_rs::{
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    println!("Running Deep Q-Learning Agent");
    type B = Autodiff<NdArray>;
    let device: &Device<B> = &Default::default();

    let mut rng = StdRng::seed_from_u64(0);
    let env = GymEnvironment::from(CartPoleEnv::new(RenderMode::None));
    let agent = CartPoleModel::<B>::init(device);
    let memory = RingbufferMemory::<Transition<CartPoleObservation, CartPoleAction>, _>::new(
        10000,
        StdRng::from_seed(rng.gen()),
    );
    let optim = AdamConfig::new().init();

    // Algorithm
    let algorithm = OffPolicyAlgorithm {
        env,
        agent,
        memory,
        optim,
        rng,
        _phantom: Default::default(),
    };

    // Execute Training Loop
    algorithm.train(1000);
}

#[derive(Clone)]
pub struct Transition<O, A> {
    pub before: O,
    pub action: A,
    pub after: O,
    pub reward: f64,
    pub done: bool,
}

pub struct OffPolicyAlgorithm<B, E, A, M, O, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: AutodiffModule<B> + Actor<O = E::O, A = E::A> + Critic<B> + Value<B>,
    M: Memory<T = Transition<E::O, E::A>, TBatch = Vec<Transition<E::O, E::A>>>,
    O: Optimizer<A, B>,
    R: Rng,
{
    env: E,
    agent: A,
    memory: M,
    optim: O,
    rng: R,
    _phantom: PhantomData<B>,
}

impl<B, E, A, M, O, R> OffPolicyAlgorithm<B, E, A, M, O, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: AutodiffModule<B>
        + Actor<O = E::O, A = E::A>
        + Critic<B, OBatch = Vec<E::O>, ABatch = Vec<E::A>>
        + Value<B, OBatch = Vec<E::O>>,
    M: Memory<T = Transition<E::O, E::A>, TBatch = Vec<Transition<E::O, E::A>>>,
    O: Optimizer<A, B>,
    R: Rng,
{
    pub fn train(mut self, n_frames: usize) {
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
            let action = self.agent.a(observation.clone());
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
            // Sample batch
            let batch = self.memory.sample_random_batch(32);
            let before = batch.iter().map(|x| x.before.clone()).collect();
            let action = batch.iter().map(|x| x.action.clone()).collect();
            let after = batch.iter().map(|x| x.after.clone()).collect();
            let reward: Vec<f64> = batch.iter().map(|x| x.reward).collect();
            let done: Vec<bool> = batch.iter().map(|x| x.done).collect();

            let reward = Tensor::from_floats(reward.as_slice(), &Default::default());
            let done = Tensor::from_floats(
                done.into_iter()
                    .map(|x| if x { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &Default::default(),
            )
            .bool();

            let pred_value_given_action_before = self.agent.q_batch(&before, &action);
            let pred_value_after = self.agent.v_batch(&after);

            let error = temporal_difference(
                reward,
                pred_value_given_action_before,
                pred_value_after,
                done,
                0.99,
            );
        }
    }
}

#[derive(Module, Debug)]
pub struct CartPoleModel<B: Backend> {
    model: MultiLayerPerceptron<B>,
}

impl<B: Backend> CartPoleModel<B> {
    pub fn init(device: &B::Device) -> Self {
        let config = MultiLayerPerceptronConfig {
            n_hidden_layers: 3,
            input_size: 4,
            hidden_size: 32,
            output_size: 2,
        };
        Self {
            model: config.init(device),
        }
    }
}

impl<B: Backend> Critic<B> for CartPoleModel<B> {
    type ABatch = Vec<CartPoleAction>;
    type OBatch = Vec<CartPoleObservation>;

    fn q_batch(&self, observations: &Self::OBatch, actions: &Self::ABatch) -> Tensor<B, 1> {
        let input: Vec<Tensor<B, 1>> = observations
            .iter()
            .map(|obs| {
                Tensor::from_floats(Vec::<f64>::from(obs.clone()).as_slice(), &self.devices()[0])
            })
            .collect();
        let input: Tensor<B, 2> = Tensor::stack(input, 0);
        let actions = actions.iter().map(|a| a.a()).collect::<Vec<usize>>();
        let actions: Tensor<B, 1, Int> = Tensor::from_ints(actions.as_slice(), &self.devices()[0]);
        let actions = actions.unsqueeze_dim(1);
        let out = self.model.forward(input);
        let out = out.gather(1, actions).squeeze(1);
        out
    }
}

impl<B: Backend> Value<B> for CartPoleModel<B> {
    type OBatch = Vec<CartPoleObservation>;

    fn v_batch(&self, observations: &Self::OBatch) -> Tensor<B, 1> {
        let input: Vec<Tensor<B, 1>> = observations
            .iter()
            .map(|obs| {
                Tensor::from_floats(Vec::<f64>::from(obs.clone()).as_slice(), &self.devices()[0])
            })
            .collect();
        let input: Tensor<B, 2> = Tensor::stack(input, 0);
        self.model.forward(input).max_dim(1).squeeze(1)
    }
}

impl<B: Backend> Actor for CartPoleModel<B> {
    type A = CartPoleAction;
    type O = CartPoleObservation;
    fn a(&self, observation: Self::O) -> Self::A {
        let input = Vec::<f64>::from(observation.clone());
        let input: Tensor<B, 1> = Tensor::from_floats(&input[0..4], &self.devices()[0]);
        let out = self.model.forward(input);
        CartPoleAction::from(
            out.into_data()
                .as_slice::<f32>()
                .unwrap()
                .into_iter()
                .enumerate()
                .max_by(|(_, a1), (_, a2)| a1.cmp(a2))
                .map(|(i, _)| i)
                .unwrap(),
        )
    }
}
