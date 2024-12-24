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

fn create_linear_layer<B: Backend>(
    input_size: usize,
    output_size: usize,
    device: &B::Device,
) -> Linear<B> {
    LinearConfig::new(input_size, output_size).init(device)
}

impl MultiLayerPerceptronConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiLayerPerceptron<B> {
        let mut linear_layers = Vec::new();
        if self.n_hidden_layers == 0 {
            linear_layers.push(create_linear_layer(
                self.input_size,
                self.output_size,
                device,
            ));
        } else {
            linear_layers.push(create_linear_layer(
                self.input_size,
                self.hidden_size,
                device,
            ));
            for _ in 0..(self.n_hidden_layers - 1) {
                linear_layers.push(create_linear_layer(
                    self.hidden_size,
                    self.hidden_size,
                    device,
                ));
            }
            linear_layers.push(create_linear_layer(
                self.hidden_size,
                self.output_size,
                device,
            ));
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

#[cfg(test)]
mod tests {

    use burn::{
        backend::NdArray,
        tensor::{Shape, Tensor},
    };

    use super::MultiLayerPerceptronConfig;

    #[test]
    fn test_multi_layer_perceptron() {
        let device = &Default::default();
        for n_hidden_layers in 0..2 {
            let output_size = 3;
            let model = MultiLayerPerceptronConfig {
                n_hidden_layers,
                input_size: 4,
                hidden_size: 32,
                output_size,
            }
            .init::<NdArray>(device);

            let x = Tensor::<NdArray, 1>::from_floats([1.0, 2.0, 3.0, 4.0], device);
            assert_eq!(model.forward(x.clone()).shape(), Shape::new([output_size]));
        }
    }
}
