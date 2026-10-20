
import random
import numpy as np
import streamlit as st

st.set_page_config(page_title="Smart Traffic Light", page_icon="🚦")

st.title("🚦 Smart Traffic Light using Q-Learning")
st.write("Train an RL agent and compare it with a fixed-timer traffic light.")

class Intersection:
    def reset(self):
        self.ns = 0
        self.ew = 0
        self.green = 0
        return self.state()

    def state(self):
        return (
            min(self.ns // 3, 4),
            min(self.ew // 3, 4),
            self.green
        )

    def step(self, action):
        self.ns += int(random.random() < 0.5)
        self.ew += int(random.random() < 0.3)

        if action == 1:
            self.green = 1 - self.green
        elif self.green == 0:
            self.ns = max(0, self.ns - 2)
        else:
            self.ew = max(0, self.ew - 2)

        reward = -(self.ns + self.ew)
        return self.state(), reward


def train(episodes=500, steps=200):
    q = np.zeros((5, 5, 2, 2))
    alpha, gamma, epsilon = 0.1, 0.9, 1.0
    rewards = []

    for episode in range(episodes):
        env = Intersection()
        state = env.reset()
        episode_reward = 0

        for _ in range(steps):
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(q[state]))

            next_state, reward = env.step(action)

            q[state][action] += alpha * (
                reward + gamma * np.max(q[next_state])
                - q[state][action]
            )

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        epsilon = max(0.05, epsilon * 0.99)

    return q, rewards


def evaluate(q, use_timer=False, steps=1000):
    env = Intersection()
    state = env.reset()
    total_reward = 0

    for t in range(steps):
        if use_timer:
            action = 1 if t % 10 == 0 else 0
        else:
            action = int(np.argmax(q[state]))

        state, reward = env.step(action)
        total_reward += reward

    return total_reward / steps


if st.button("Train and Test"):
    with st.spinner("Training Q-Learning agent..."):
        q, rewards = train()

        random.seed(42)
        rl_score = evaluate(q)
        random.seed(42)
        timer_score = evaluate(q, use_timer=True)

    st.subheader("Results")
    col1, col2 = st.columns(2)

    col1.metric("RL Average Reward", round(rl_score, 2))
    col2.metric("Timer Average Reward", round(timer_score, 2))

    st.subheader("Training Reward Curve")
    st.line_chart(rewards)

    if rl_score > timer_score:
        st.success("RL performed better in this test.")
    else:
        st.info("The fixed timer performed better in this test.")

    st.caption(
        "Results are from a simplified simulation. "
        "Repeat tests with multiple random seeds for a fair comparison."
    )
