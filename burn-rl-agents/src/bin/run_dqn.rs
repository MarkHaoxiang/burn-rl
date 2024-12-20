use burn::prelude::*;
use burn_rl::{
    environment::gym_rs::GymEnvironment,
    module::{
        abc::Critic,
        nn::multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
    },
};
use gym_rs::{
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
    utils::renderer::RenderMode,
};

fn main() {
    println!("Running Deep Q-Learning Agent");

    // Define the environment
    let env = GymEnvironment::from(CartPoleEnv::new(RenderMode::None));
    // Define the model
}

#[derive(Module, Debug)]
pub struct CartPoleCritic<B: Backend> {
    model: MultiLayerPerceptron<B>,
}

impl<B: Backend> CartPoleCritic<B> {
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

impl<B: Backend> Critic for CartPoleCritic<B> {
    type Action = usize;
    type Observation = CartPoleObservation;

    fn q(&self, observation: &Self::Observation, action: &Self::Action) -> f64 {
        let input = Vec::<f64>::from(observation.clone());
        let input: Tensor<B, 1> = Tensor::from_floats(&input[0..4], &self.devices()[0]);
        let out = self.model.forward(input);
        out.into_data().as_slice::<f64>().unwrap()[*action]
    }
}
