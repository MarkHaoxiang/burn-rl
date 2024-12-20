use std::fmt::Debug;

use burn::{prelude::*, tensor::TensorKind};

pub struct DeepQNetworkLossConfig {
    discount_factor: f64,
}

impl DeepQNetworkLossConfig {
    pub fn init(&self) -> DeepQNetworkLoss {
        self.assertions();
        DeepQNetworkLoss {
            discount_factor: self.discount_factor,
        }
    }

    fn assertions(&self) {
        assert!(
            0.0 <= self.discount_factor && self.discount_factor <= 1.0,
            "The discount factor should be in the interval [0,1]. got {}",
            self.discount_factor
        )
    }
}

#[derive(Module, Clone, Debug)]
pub struct DeepQNetworkLoss {
    discount_factor: f64,
}

impl DeepQNetworkLoss {
    pub fn forward<B: Backend, const D: usize, A: TensorKind<B>, O: TensorKind<B>>(
        &self,
        before: Tensor<B, D, O>,
        action: Tensor<B, D, A>,
        after: Tensor<B, D, O>,
        done: Tensor<B, D, Bool>,
    ) {
        todo!()
    }
}
