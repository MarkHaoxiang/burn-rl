use burn::backend::{Autodiff, CudaJit, NdArray, Wgpu};
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
struct DQNData<B: Backend>(
    Tensor<B, 2>,
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 1>,
    Tensor<B, 1, Bool>,
);

fn prepare_dqn_data<B: Backend>(device: &Device<B>, batch_size: usize) -> DQNData<B> {
    let before = Tensor::random([batch_size, 32], Distribution::Default, device);
    let action =
        Tensor::<B, 2, Int>::random([batch_size, 1], Distribution::Uniform(0.0, 4.0), device);
    let after = Tensor::random([batch_size, 32], Distribution::Default, device);
    let reward = Tensor::random([batch_size], Distribution::Default, device);
    let done = Tensor::<B, 1, Int>::random([batch_size], Distribution::Uniform(0.0, 1.0), device);
    let done = done.bool();
    DQNData(before, action, after, reward, done)
}

fn prepare_dqn<B: Backend>(
    batch_size: usize,
) -> (DeepQNetworkLoss, WithTarget<B, DQNModel<B>>, DQNData<B>) {
    let loss = DeepQNetworkLossConfig::new().init();
    let model_config = MultiLayerPerceptronConfig::new([32, 128, 128, 5].to_vec());

    let device: &Device<B> = &Default::default();
    let model = WithTarget::<B, DQNModel<B>>::new(DQNModel {
        model: model_config.init(device),
    });
    let data: DQNData<B> = prepare_dqn_data(device, batch_size);
    (loss, model, data)
}

fn dqn_forward<B: Backend>(
    loss: &DeepQNetworkLoss,
    model: &WithTarget<B, DQNModel<B>>,
    data: &DQNData<B>,
) -> Tensor<B, 1> {
    let result = loss.forward(
        model,
        &data.0,
        &data.1,
        &data.2,
        data.3.clone(),
        data.4.clone(),
        Mean,
    );
    result
}

fn dqn_backward<B: AutodiffBackend>(
    loss: &DeepQNetworkLoss,
    model: &WithTarget<B, DQNModel<B>>,
    data: &DQNData<B>,
) -> <B as AutodiffBackend>::Gradients {
    let result = loss.forward(
        model,
        &data.0,
        &data.1,
        &data.2,
        data.3.clone(),
        data.4.clone(),
        Mean,
    );
    result.backward()
}

pub fn dqn_forward_ndarray(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<NdArray>(64);
    c.bench_function("dqn ndarray forward 64", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });

    let (loss, model, data) = prepare_dqn::<NdArray>(512);
    c.bench_function("dqn ndarray forward 512", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });
}

pub fn dqn_forward_cuda(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<CudaJit>(64);
    c.bench_function("dqn cuda forward 64", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });

    let (loss, model, data) = prepare_dqn::<CudaJit>(512);
    c.bench_function("dqn cuda forward 512", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });
}

pub fn dqn_forward_wgpu(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<Wgpu>(64);
    c.bench_function("dqn wgpu forward 64", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });

    let (loss, model, data) = prepare_dqn::<Wgpu>(512);
    c.bench_function("dqn wgpu forward 512", |b| {
        b.iter(|| dqn_forward(&loss, &model, &data))
    });
}

pub fn dqn_backward_ndarray(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<Autodiff<NdArray>>(64);
    c.bench_function("dqn ndarray backward 64", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });

    let (loss, model, data) = prepare_dqn::<Autodiff<NdArray>>(512);
    c.bench_function("dqn ndarray backward 512", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });
}

pub fn dqn_backward_cuda(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<Autodiff<CudaJit>>(64);
    c.bench_function("dqn cuda backward 64", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });
    let (loss, model, data) = prepare_dqn::<Autodiff<CudaJit>>(512);
    c.bench_function("dqn cuda backward 512", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });
}

pub fn dqn_backward_wgpu(c: &mut Criterion) {
    let (loss, model, data) = prepare_dqn::<Autodiff<Wgpu>>(64);
    c.bench_function("dqn wgpu backward 64", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });
    let (loss, model, data) = prepare_dqn::<Autodiff<Wgpu>>(512);
    c.bench_function("dqn wgpu backward 512", |b| {
        b.iter(|| dqn_backward(&loss, &model, &data))
    });
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default();
    targets = dqn_backward_ndarray, dqn_backward_cuda, dqn_backward_wgpu, dqn_forward_ndarray, dqn_forward_cuda, dqn_forward_wgpu
}
criterion_main!(benches);
