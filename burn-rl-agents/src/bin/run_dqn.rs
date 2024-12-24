use std::marker::PhantomData;

use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamConfig, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_rl::{
    data::memory::{Memory, RingbufferMemory},
    environment::{
        gym_rs::{CartPoleAction, GymEnvironment},
        Environment, Space,
    },
    module::{
        abc::{Actor, Critic},
        nn::multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
    },
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
    let agent = CartPoleModel::<B>::new();
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
    A: AutodiffModule<B> + Actor<O = E::O, A = E::A> + Critic<B>,
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
    A: AutodiffModule<B> + Actor<O = E::O, A = E::A> + Critic<B>,
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
            observation = next_observation;
        }
    }
}

#[derive(Module, Debug)]
pub struct CartPoleModel<B: Backend> {
    model: MultiLayerPerceptron<B>,
}

impl<B: Backend> CartPoleModel<B> {
    pub fn new() -> Self {
        let config = MultiLayerPerceptronConfig {
            n_hidden_layers: 3,
            input_size: 4,
            hidden_size: 32,
            output_size: 2,
        };
        Self {
            model: config.init(),
        }
    }
}

impl<B: Backend> Critic<B> for CartPoleModel<B> {
    type ABatch = Vec<usize>;
    type OBatch = Vec<CartPoleObservation>;

    fn q_batch(&self, observations: &Self::OBatch, actions: &Self::ABatch) -> Tensor<B, 1> {
        let input: Vec<Tensor<B, 1>> = observations
            .iter()
            .map(|obs| {
                Tensor::from_floats(Vec::<f64>::from(obs.clone()).as_slice(), &self.devices()[0])
            })
            .collect();
        let input = Tensor::stack(input, 0);
        let actions = Tensor::from_ints(actions.as_slice(), &self.devices()[0]);
        let out = self.model.forward(input).gather(1, actions);
        out
    }
}

impl<B: Backend> Actor for CartPoleModel<B> {
    type A = CartPoleAction;
    type O = CartPoleObservation;
    fn a(&self, observation: Self::O) -> Self::A {
        todo!()
    }
}
