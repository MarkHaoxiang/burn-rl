use burn::{
    prelude::Backend,
    tensor::{Bool, Tensor},
};

pub fn temporal_difference<B: Backend>(
    reward: Tensor<B, 1>,
    pred_value_given_action_before: Tensor<B, 1>,
    pred_value_after: Tensor<B, 1>,
    done: Tensor<B, 1, Bool>,
    discount_factor: f64,
) -> Tensor<B, 1> {
    let trajectory_value_before = reward + done.float() * pred_value_after * discount_factor;
    pred_value_given_action_before - trajectory_value_before
}
