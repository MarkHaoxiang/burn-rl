use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::Module;
use burn::nn::loss::Reduction::Mean;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;
use burn::tensor::Distribution;
use burn::{
    prelude::Backend,
    tensor::{Bool, Int, Tensor},
};
use burn_rl::{
    module::{
        component::{Critic, Value},
        nn::{
            multi_layer_perceptron::{MultiLayerPerceptron, MultiLayerPerceptronConfig},
            target_model::WithTarget,
        },
    },
    objective::dqn::{DeepQNetworkLoss, DeepQNetworkLossConfig},
};
use criterion::{criterion_group, criterion_main, Criterion};

#[derive(Module, Debug)]
struct DQNModel<B: Backend> {
    model: MultiLayerPerceptron<B>,
} // TODO: This should really be part of the library and tested
impl<B: Backend> Value<B> for DQNModel<B> {
    type OBatch = Tensor<B, 2>;

    fn v_batch(&self, observations: &Self::OBatch) -> Tensor<B, 1> {
        self.model
            .forward(observations.clone())
            .max_dim(1)
            .squeeze(1)
    }
}
impl<B: Backend> Critic<B> for DQNModel<B> {
    type OBatch = Tensor<B, 2>;

    type ABatch = Tensor<B, 2, Int>;

    fn q_batch(&self, observations: &Self::OBatch, actions: &Self::ABatch) -> Tensor<B, 1> {
        self.model
            .forward(observations.clone())
            .gather(1, actions.clone())
            .squeeze(1)
    }
}

fn prepare_dqn_data<B: Backend>(
    device: &Device<B>,
    batch_size: usize,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 1>,
    Tensor<B, 1, Bool>,
) {
    let before = Tensor::random([batch_size, 4], Distribution::Default, device);
    let action =
        Tensor::<B, 2, Int>::random([batch_size, 1], Distribution::Uniform(0.0, 1.0), device);
    let after = Tensor::random([batch_size, 4], Distribution::Default, device);
    let reward = Tensor::random([batch_size], Distribution::Default, device);
    let done = Tensor::<B, 1, Int>::random([batch_size], Distribution::Uniform(0.0, 1.0), device);
    let done = done.bool();
    (before, action, after, reward, done)
}

fn dqn_fn<B: AutodiffBackend>(
    loss: &DeepQNetworkLoss,
    model: &WithTarget<B, DQNModel<B>>,
    before: &Tensor<B, 2>,
    action: &Tensor<B, 2, Int>,
    after: &Tensor<B, 2>,
    reward: Tensor<B, 1>,
    done: Tensor<B, 1, Bool>,
) {
    loss.forward(model, before, action, after, reward, done, Mean);
}

pub fn dqn_benchmark(c: &mut Criterion) {
    let loss = DeepQNetworkLossConfig::new().init();
    let model_config = MultiLayerPerceptronConfig::new([4, 32, 32, 2].to_vec());

    // NdArray
    type B1 = Autodiff<NdArray>;
    let device: &Device<B1> = &Default::default();
    let model = WithTarget::<B1, DQNModel<B1>>::new(DQNModel {
        model: model_config.init(device),
    });
    let (before, action, after, reward, done) = prepare_dqn_data(device, 32);

    c.bench_function("dqn ndarray", |b| {
        b.iter(|| {
            dqn_fn(
                &loss,
                &model,
                &before,
                &action,
                &after,
                reward.clone(),
                done.clone(),
            )
        })
    });

    // Wgpu
    type B2 = Autodiff<Wgpu>;
    let device: &Device<B2> = &WgpuDevice::BestAvailable;
    let model = WithTarget::<B2, DQNModel<B2>>::new(DQNModel {
        model: model_config.init(device),
    });
    let (before, action, after, reward, done) = prepare_dqn_data(device, 32);

    c.bench_function("dqn wgpu", |b| {
        b.iter(|| {
            dqn_fn(
                &loss,
                &model,
                &before,
                &action,
                &after,
                reward.clone(),
                done.clone(),
            )
        })
    });
}

criterion_group!(benches, dqn_benchmark);
criterion_main!(benches);
