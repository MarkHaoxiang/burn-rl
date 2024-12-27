use std::fmt::Debug;

use burn::{module::AutodiffModule, prelude::*, tensor::backend::AutodiffBackend};
use nn::loss::Reduction;

use crate::module::{
    component::{Critic, Value},
    nn::target_model::WithTarget,
};

use super::temporal_difference::temporal_difference;

#[derive(Config)]
pub struct DeepQNetworkLossConfig {
    #[config(default = 0.99)]
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
    pub fn forward<B, M, OBatch, ABatch>(
        &self,
        model: &WithTarget<B, M>,
        before: &OBatch,
        action: &ABatch,
        after: &OBatch,
        reward: Tensor<B, 1>,
        done: Tensor<B, 1, Bool>,
        reduction: Reduction,
    ) -> Tensor<B, 1>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>
            + Critic<B, OBatch = OBatch, ABatch = ABatch>
            + Value<B, OBatch = OBatch>,
    {
        let pred_value_given_action_before = model.model.q_batch(before, action);
        let pred_value_after = model.target.v_batch(after);

        let tensor = temporal_difference(
            reward,
            pred_value_given_action_before,
            pred_value_after,
            done,
            self.discount_factor,
        )
        .powf_scalar(2.0);

        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
        }
    }
}
