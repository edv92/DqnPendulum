reward = 1
q_now = 0.5
discount = 0.9
learning_rate = 0.2
q_future = 0.5
for i in range(1100):
    q_now = q_now + learning_rate*(reward + discount*q_future - q_now)
    q_future = q_now
    print(f"q_now for iteration number {i} is: {q_now}")