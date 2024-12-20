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

impl<B: Backend> Critic<B> for CartPoleCritic<B> {
    type ActionBatch = Vec<usize>;
    type ObservationBatch = Vec<CartPoleObservation>;

    fn q_batch(
        &self,
        observations: &Self::ObservationBatch,
        actions: &Self::ActionBatch,
    ) -> Tensor<B, 1> {
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
