use rand::{seq::IteratorRandom, Rng};
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub trait Memory {
    type T;
    type TBatch;

    fn push(&mut self, value: Self::T);

    fn append(&mut self, values: Self::TBatch);

    fn sample_random_batch(&mut self, n: usize) -> Self::TBatch;
}

pub struct RingbufferMemory<T: Clone, R: Rng> {
    rng: R,
    buffer: AllocRingBuffer<T>,
}

impl<T: Clone, R: Rng> RingbufferMemory<T, R> {
    pub fn new(capacity: usize, rng: R) -> RingbufferMemory<T, R> {
        RingbufferMemory {
            rng,
            buffer: AllocRingBuffer::new(capacity),
        }
    }
}

impl<T: Clone, R: Rng> Memory for RingbufferMemory<T, R> {
    type T = T;
    type TBatch = Vec<T>;

    fn push(&mut self, value: T) {
        self.buffer.push(value);
    }

    fn append(&mut self, values: Self::TBatch) {
        for value in values {
            self.push(value);
        }
    }

    fn sample_random_batch(&mut self, n: usize) -> Vec<T> {
        self.buffer
            .iter()
            .choose_multiple(&mut self.rng, n)
            .into_iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use rand::{rngs::StdRng, SeedableRng};

    use super::{Memory, RingbufferMemory};

    #[test]
    fn test_ringbuffer_memory() {
        let rng = StdRng::seed_from_u64(1234);
        let mut memory = RingbufferMemory::new(10, rng);
        for i in 0..10 {
            memory.push(i);
        }
        let sample = memory.sample_random_batch(5);
        let expected = expect![[r#"
            [
                6,
                7,
                2,
                3,
                4,
            ]
        "#]];
        expected.assert_debug_eq(&sample);
        for i in 10..15 {
            memory.push(i);
        }
        let sample = memory.sample_random_batch(5);
        let expected = expect![[r#"
            [
                13,
                6,
                10,
                8,
                11,
            ]
        "#]];
        expected.assert_debug_eq(&sample);
    }
}
