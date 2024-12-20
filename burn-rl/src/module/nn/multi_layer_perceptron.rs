use std::fmt::Debug;

use burn::module::Module;
use burn::nn::Linear;
use burn::prelude::*;
use nn::{LeakyRelu, LeakyReluConfig, LinearConfig};

pub struct MultiLayerPerceptronConfig {
    pub n_hidden_layers: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

#[derive(Module, Debug)]
pub struct MultiLayerPerceptron<B: Backend> {
    linear_layers: Vec<Linear<B>>,
    activation: LeakyRelu,
}

fn create_linear_layer<B: Backend>(input_size: usize, output_size: usize) -> Linear<B> {
    LinearConfig::new(input_size, output_size).init(&Default::default())
}

impl MultiLayerPerceptronConfig {
    pub fn init<B: Backend>(&self) -> MultiLayerPerceptron<B> {
        let mut linear_layers = Vec::new();
        if self.n_hidden_layers == 0 {
            linear_layers.push(create_linear_layer(self.input_size, self.output_size));
        } else {
            linear_layers.push(create_linear_layer(self.input_size, self.output_size));
            for _ in 0..(self.n_hidden_layers - 1) {
                linear_layers.push(create_linear_layer(self.input_size, self.output_size));
            }
            linear_layers.push(create_linear_layer(self.input_size, self.output_size));
        }
        let activation = LeakyReluConfig::new().init();
        MultiLayerPerceptron {
            linear_layers,
            activation,
        }
    }
}

impl<B: Backend> MultiLayerPerceptron<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = input;

        match self.linear_layers.last() {
            Some(f) => {
                for layer in self.linear_layers[..self.linear_layers.len() - 1].iter() {
                    x = layer.forward(x);
                    x = self.activation.forward(x);
                }
                f.forward(x)
            }
            None => x,
        }
    }
}
