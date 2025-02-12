use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    prelude::*,
    tensor::ElementComparison,
};
use burn_rl::{
    data::{memory::RingbufferMemory, util::Transition},
    environment::gym_rs::{CartPoleAction, GymEnvironment},
    module::{
        component::{Actor, Critic, Value},
        nn::multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
    },
};
use burn_rl_agents::{dqn::DeepQNetworkAgentConfig, off_policy::OffPolicyAlgorithmConfig};
use gym_rs::{
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    println!("Running Deep Q-Learning Agent");
    type B = Autodiff<NdArray>;
    // type B = Autodiff<Wgpu>;
    let device: &Device<B> = &Default::default();
    let mut rng = StdRng::seed_from_u64(30000);

    // Construct environment
    let env = GymEnvironment::from(CartPoleEnv::new(RenderMode::None));
    let eval_env = GymEnvironment::from(CartPoleEnv::new(RenderMode::None));

    // Construct agent
    let dqn_model = CartPoleModel::<B>::init(device);
    let optim = AdamConfig::new().init();
    let agent = DeepQNetworkAgentConfig::new().init(dqn_model, optim);

    // Construct memory
    let memory = RingbufferMemory::<Transition<CartPoleObservation, CartPoleAction>, _>::new(
        20_000,
        StdRng::from_seed(rng.gen()),
    );

    // Algorithm
    let algorithm =
        OffPolicyAlgorithmConfig::new(20_000, 128).init(env, eval_env, agent, memory, rng);

    // Execute Training Loop
    algorithm.train();
}

#[derive(Module, Debug)]
pub struct CartPoleModel<B: Backend> {
    model: MultiLayerPerceptron<B>,
}

impl<B: Backend> CartPoleModel<B> {
    pub fn init(device: &B::Device) -> Self {
        let sizes = [4, 64, 64, 2].to_vec();
        let config = MultiLayerPerceptronConfig::new(sizes);
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
