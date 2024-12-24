use crate::environment::Environment;

pub struct Transition<E: Environment> {
    pub before: E::O,
    pub action: E::A,
    pub after: E::O,
    pub reward: f64,
    pub done: bool,
}

impl<E: Environment> Clone for Transition<E>
where
    E::O: Clone,
    E::A: Clone,
{
    fn clone(&self) -> Self {
        Self {
            before: self.before.clone(),
            action: self.action.clone(),
            after: self.after.clone(),
            reward: self.reward,
            done: self.done,
        }
    }
}

impl<E: Environment> Transition<E> {
    pub fn to_nested_tuple(self) -> (E::O, (E::A, (E::O, (f64, bool)))) {
        (
            self.before,
            (self.action, (self.after, (self.reward, self.done))),
        )
    }
}

pub fn collect_multiple<E: Environment, P: FnMut(&E::O) -> E::A>(
    env: &mut E,
    observation: Option<E::O>,
    policy: &mut P,
    n_steps: usize,
) -> Vec<Transition<E>> {
    let mut before = match observation {
        Some(observation) => observation,
        None => env.reset(),
    };
    let mut result = Vec::new();
    for _ in 0..n_steps {
        let action = policy(&before);
        let (after, reward, done) = env.step(action.clone());
        result.push(Transition {
            before,
            action,
            after: after.clone(),
            reward,
            done,
        });
        before = match done {
            true => env.reset(),
            false => after,
        };
    }
    result
}

pub fn collect_single<E: Environment, P: FnMut(&E::O) -> E::A>(
    env: &mut E,
    observation: Option<E::O>,
    policy: &mut P,
) -> Transition<E> {
    collect_multiple(env, observation, policy, 1)
        .into_iter()
        .nth(0)
        .unwrap()
}
