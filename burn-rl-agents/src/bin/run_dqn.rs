use std::marker::PhantomData;

use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, ElementComparison},
};
use burn_rl::{
    data::{
        memory::{Memory, RingbufferMemory},
        util::{collect_multiple, collect_single, Transition},
    },
    environment::{
        gym_rs::{CartPoleAction, GymEnvironment},
        Environment, Space,
    },
    module::{
        component::{Actor, Critic, Value},
        nn::{
            multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
            target_model::WithTarget,
        },
    },
    objective::dqn::DeepQNetworkLossConfig,
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
    let agent = WithTarget::new(agent);

    let memory = RingbufferMemory::<Transition<GymEnvironment<CartPoleEnv>>, _>::new(
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
    algorithm.train();
}

pub struct OffPolicyAlgorithm<B, E, A, M, O, R>
where
    E: Environment,
    B: AutodiffBackend,
    A: AutodiffModule<B>
        + Actor<O = E::O, A = E::A>
        + Value<B, OBatch = Vec<E::O>>
        + Critic<B, OBatch = Vec<E::O>, ABatch = Vec<E::A>>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    O: Optimizer<A, B>,
    R: Rng,
{
    env: E,
    agent: WithTarget<B, A>,
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
        + Value<B, OBatch = Vec<E::O>>
        + Critic<B, OBatch = Vec<E::O>, ABatch = Vec<E::A>>,
    M: Memory<T = Transition<E>, TBatch = Vec<Transition<E>>>,
    O: Optimizer<A, B>,
    R: Rng,
{
    pub fn train(mut self) {
        // Early Start
        let mut policy = |_: &E::O| <E::A>::sample(&mut self.rng);
        let transitions = collect_multiple(&mut self.env, None, &mut policy, 1000);
        let _ = transitions.into_iter().map(|x| self.memory.push(x));

        let dqn = DeepQNetworkLossConfig {
            discount_factor: 0.99,
        }
        .init();

        // Main Training Loop
        let mut observation = self.env.reset();
        for _ in 0..1000 {
            // Step Environment
            let transition = collect_single(&mut self.env, Some(observation), &mut |o| {
                self.agent.model.a(o)
            });
            observation = transition.after.clone();
            self.memory.push(transition);

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

            let loss = dqn.forward(
                &self.agent,
                &before,
                &action,
                &after,
                reward,
                done,
                nn::loss::Reduction::Mean,
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.agent.model);
            self.agent.model = self.optim.step(0.003, self.agent.model, grads);
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
    fn a(&self, observation: &Self::O) -> Self::A {
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
