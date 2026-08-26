import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 9: Practical - Basics of Reinforcement Learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the year 2147, Earth’s surface has become a patchwork of unstable terrains after decades of climate decay.
    Our AI-controlled exploration unit — Bot-7 — is dispatched from its landing pod (🔵) to reach a high-priority extraction point (🏁).

    The environment is treacherous and energy-limited. Bot-7 must autonomously learn the best route across a grid of dynamic terrain:

    🌱 Biofields: Soft, moss-covered terrain. Easy to traverse.

    🌊 Flooded Zones: Shallow but energy-draining water channels.

    ⛰️ Crater Ridges: Volcanic rubble requiring intense power to cross.

    Each grid cell represents one unit of terrain. Moving across terrain consumes energy — some more than others.

    *Source: ChatGPT 4o (prompt: shortest path algorithm story with a sci-fi theme)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import random

    class GridGame:
        TERRAIN_TYPES = {
            "🌱": 1,  # Land
            "🌊": 3,  # Sea
            "⛰️": 5   # Mountain
        }

        def __init__(self, rows, cols, terrain_weights=None):
            """
            :param rows: Number of rows in the grid
            :param cols: Number of columns in the grid
            :param terrain_weights: Optional weights for each terrain type
            """
            self.rows = rows
            self.cols = cols
            self.terrain_symbols = list(self.TERRAIN_TYPES.keys())
            self.terrain_costs = self.TERRAIN_TYPES

            self.grid = self.generate_random_grid(rows, cols, terrain_weights)

            self.cost_input_grid = {
        (i,j): mo.ui.number(value = None) for i in range(rows) for j in range(cols) }

            self.value_input_grid = {
        (i,j): mo.ui.number(value = None) for i in range(rows) for j in range(cols) }

            self.start, self.goal = self.pick_start_and_goal()
            self.trajectory = []

        def generate_random_grid(self, rows, cols, weights=None):
            """Generates a random terrain grid."""
            if weights is None:
                weights = [1] * len(self.terrain_symbols)  # uniform if not provided
            return [
                [random.choices(self.terrain_symbols, weights=weights, k=1)[0] for _ in range(cols)]
                for _ in range(rows)
            ]

        def pick_start_and_goal(self):
            """Randomly selects non-overlapping start and goal cells."""
            all_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)]
            start = random.choice(all_cells)
            all_cells.remove(start)
            goal = random.choice(all_cells)
            return start, goal

        def in_bounds(self, coord):
            x, y = coord
            return 0 <= x < self.rows and 0 <= y < self.cols

        def get_cost(self, coord):
            if not self.in_bounds(coord):
                raise ValueError(f"Coordinate {coord} is out of bounds.")
            terrain = self.grid[coord[0]][coord[1]]
            return self.terrain_costs.get(terrain, float("inf"))

        def evaluate_trajectory(self, trajectory):
            total_cost = 0
            for coord in trajectory:
                total_cost += self.get_cost(coord)
                if coord == self.goal:
                    return total_cost, True
            return total_cost, False

        def evaluate_policy(self, policy, steps):
            if not self.start:
                raise ValueError("Start state must be defined.")

            actions = {
                "up":    (-1, 0),
                "down":  (1, 0),
                "left":  (0, -1),
                "right": (0, 1)
            }

            trajectory = []
            current = self.start
            total_cost = self.get_cost(current)

            for _ in range(steps):
                if current == self.goal:
                    return total_cost, True

                action = policy(current)
                move = actions.get(action)
                if move is None:
                    raise ValueError(f"Invalid action '{action}' returned by policy.")
                next_state = (current[0] + move[0], current[1] + move[1])
                if not self.in_bounds(next_state):
                    break
                current = next_state
                trajectory.append(current)
                total_cost += self.get_cost(current)

            return trajectory, total_cost, current == self.goal

        def print_values(self, values, input_grid, test=False):
            def color_style(value, color):
                return f"<span style='background-color:{color}; color:white; padding:2px 6px'>{value}</span>"

            def color_cell(index, value):
                if (self.goal and index == self.goal):
                    return color_style("🏁", "black")
                if (self.start and index == self.start):
                    return color_style("🔵", "blue")  
                if test:
                    return color_style(value, "white") + f"{input_grid[index]}"
                else:
                    if input_grid[index].value == None:
                        return ""
                    if input_grid[index].value == values[index]:
                        return color_style( input_grid[index].value, "green")
                    else:
                        return color_style( input_grid[index].value, "red")                
            # Format rows
            rows = []
            for i, row in enumerate(self.grid):
                formatted_row = [color_cell((i, j), val) for j, val in enumerate(row)]
                rows.append("| " + " | ".join(formatted_row) + " |")

            # Markdown table header separator
            header_separator = "| " + " | ".join(["---"] * len(self.grid[0])) + " |"
            header = "| " + " | ".join([" " for i in range(len(self.grid[0]))]) + " |"
            # Insert header separator after first row
            return "\n".join([header, header_separator] + rows)

        def print_grid(self, trajectory = None):
            def color_style(value, color):
                return f"<span style='background-color:{color}; color:white; padding:2px 6px'>{value}</span>"

            def color_cell(index, value):
                if (self.goal and index == self.goal):
                    return color_style("🏁", "black")
                if (self.start and index == self.start):
                    return color_style("🔵", "blue")  
                if (trajectory and (index in trajectory)):
                    return color_style(value, "green")
                else:
                    return color_style(value, "white")

            # Format rows
            rows = []
            for i, row in enumerate(self.grid):
                formatted_row = [color_cell((i, j), val) for j, val in enumerate(row)]
                rows.append("| " + " | ".join(formatted_row) + " |")

            # Markdown table header separator
            header_separator = "| " + " | ".join(["---"] * len(self.grid[0])) + " |"
            header = "| " + " | ".join([" " for i in range(len(self.grid[0]))]) + " |"
            # Insert header separator after first row
            return "\n".join([header, header_separator] + rows)

    return (GridGame,)


@app.cell
def _(GridGame):
    game = GridGame(rows=5, cols=5)
    return (game,)


@app.cell(hide_code=True)
def _(game, mo):
    mo.md(f"""
    {game.print_grid()}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the terrain map is known, Bot-7 can compute the most energy-efficient route to the target using principles of **dynamic programming** — evaluating each cell's cumulative cost from goal to start and choosing the least expensive path. In particular, let's assign the following costs to going through each type of terrain: "🌱": 1, "🌊": 3, "⛰️": 5. Below is a solution to this problem using a very well known algorithm known as Dijkstra's Algorithm (feel free to look at the code if you are interested, but in reinforcement learning our goal will be to understand how to do this without knowing the map).
    """)
    return


@app.cell(hide_code=True)
def _():
    import heapq

    # Note this is an artificial example that calculates the exact costs and rewards to be used in the remainder of the notebook for demonstration purposes. It combines both cost based calculations and reward value function calculation instead of relying on only one of the approaches.

    def dijkstra(grid, terrain_costs, start, goal):
        """
        Finds the lowest-cost path using Dijkstra's algorithm.

        :param grid: 2D list of terrain symbols (e.g. [["🌱", "🌊", "⛰️"]])
        :param terrain_costs: dict mapping symbols to cost (e.g. {"🌱": 1, "🌊": 3, "⛰️": 5})
        :param start: (row, col) tuple
        :param goal: (row, col) tuple
        :return: (total_cost, path as list of (row, col))
        """
        rows, cols = len(grid), len(grid[0])

        def in_bounds(coord):
            x, y = coord
            return 0 <= x < rows and 0 <= y < cols

        def get_cost(coord):
            x, y = coord
            return terrain_costs.get(grid[x][y], float("inf"))

        def get_neighbors(coord):
            x, y = coord
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            return [(x + dx, y + dy) for dx, dy in directions if in_bounds((x + dx, y + dy))]

        visited = set()
        heap = [(0, goal, [goal])]

        costs = { (i, j): float('inf') for i in range(rows) for j in range(cols) }
        costs[goal] = 0

        path_to_goal = []
        cost_to_goal = float("inf")

        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == start:
                path_to_goal = path
                cost_to_goal = cost

            for neighbor in get_neighbors(current):
                new_cost = cost + get_cost(neighbor)
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor, [neighbor] + path))

        return cost_to_goal, path_to_goal, costs

    return (dijkstra,)


@app.cell
def _(dijkstra, game):
    cost, trajectory, costs = dijkstra(game.grid, game.terrain_costs, game.start, game.goal)
    print(f"\nTotal Cost: {cost}")
    print(f"Trajectory: {trajectory}")
    return costs, trajectory


@app.cell(hide_code=True)
def _(game, mo, trajectory):
    mo.md(f"""
    {game.print_grid(trajectory)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Dijkstra's algorithm (and more generally dynamic programming algorithms) rely on a simple intution, which will also be very useful to us later with RL.  From each cell in the grid there is a smallest cost path to the goal (not necessarily unique). The expected cost of this path is sometimes referred to as the **cost-to-go** of that state. Let's try to understand how to calculate the value function. Fill in the table below with what you think the cost of the optimal path will be (smallest total cost to the goal).
    """)
    return


@app.cell(hide_code=True)
def _(costs, game, mo):
    mo.md(f"""
    {game.print_values(costs, game.cost_input_grid, test=True)}
    """)
    return


@app.cell(hide_code=True)
def _(costs, game, mo):
    button = mo.ui.button(label="Verify Optimal Costs", value="", kind="success", on_click= lambda value : game.print_values(costs, game.cost_input_grid))
    button
    return (button,)


@app.cell(hide_code=True)
def _(button, mo):
    mo.md(f"""
    {button.value}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Before we proceed further, we’ll make one important conceptual shift. From this point on, we will refer not to costs, but to rewards.


    - In reinforcement learning, agents are trained to maximize reward rather than minimize cost.

    - Mathematically, the two are equivalent: *reward* = −*cost*

    So when Bot-7 is faced with a grid of terrain, each movement will now yield a (negative) reward, representing energy loss. Its mission becomes one of **maximizing total reward**, which naturally leads to minimizing total energy use.

    This reward-based framing aligns with how most RL algorithms are formulated, and sets the stage for what comes next: value functions, policies, and learning through experience.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's now frame the problem as an RL problem and understand better the terms: **agent**, **environment**, **state**, **action**, **reward** as applied to this example.

    The **agent** is simply our Bot-7 (or its decision making system). Its **state** is its position in our grid matrix and can be described with coordinates (i,j) representing the row and column of the position. The **actions** available to Bot-7 are moving up, down, left, or right (except when it is at the boundary, then only some of the actions are available).

    The **environment** abstracts all the complexities of putting the agent in the next state based on its action decision. This could be the robot actuators, the effects of wind and anything else.  For this example we'll assume a deterministic environment that simply maps the current state and action to the next state:

    $$s(t+1) = f(s(t), a(t))$$

    by moving the robot to up, down, left, or right based on the action $a(t)$.

    Note: to make things more interesting we could introduce a stochastic environment that, for example, moves the bot to the state corresponding to the action with 80% probability, and moves it to one of the other neighboring states with a 20% probability.

    Finally, along with updating the state, the environment also assigns some **reward** (in our case corresponding to one of the terrain types the bot ended up in or an extra reward if it reached the goal)

    Rewards:

    - 🌱: -1
    - 🌊: -3
    - ⛰️: -5

    Recall that in RL the agent usually does not have access to the map and doesn't know in advance which cells will give which rewards. It will have to discover it by trying various actions in an efficient way.
    """)
    return


@app.cell(hide_code=True)
def _(game, mo):
    mo.md(f"""
    For now, let's pretend again that we know what the map looks like:

    {game.print_grid()}

    The agent needs some way to make decisions based on the information it has. This is called a **policy**. Let's assume the agent's policy is to first move vertically to get to the same column as the goal and then move horizontally. Let's evaluate this policy to calculate the total amount of reward the agent would get.
    """)
    return


@app.cell
def _(game):
    # simple policy returning the direction to move 
    # horizontally (+-1, 0) or vertically (0, +-1)
    def simple_policy(state):
        x, y = state
        gx, gy = game.goal
        if (x < gx):
            return (1, 0)
        if (x > gx):
            return (-1, 0)
        if (y < gy):
            return (0, 1)
        if (y > gy):
            return (0, -1)

    # Function to get the reward of moving to a new cell
    # This is part of the environment and is used only
    # to evaluate the policy, not to learn it.
    def get_reward(x, y):
        return -game.terrain_costs.get(game.grid[x][y])

    # Function to take a step in the environment
    # This simulates the environment's response to the agent's action
    # and returns the new state and the reward
    def step(state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        return (nx, ny), get_reward(nx, ny)

    # Function to evaluate the policy by simulating the agent's actions
    # based on the policy we define, in this case: simple_policy
    def evaluate_policy():
        state = game.start
        trajectory = []
        total_reward = 0
        while state != game.goal:
            action = simple_policy(state)
            state, reward = step(state, action)
            if state != game.goal:
                total_reward += reward
                trajectory.append(state)
        print(f"\nTotal reward: {total_reward}")
        print(f"Trajectory: {trajectory}")
        return trajectory

    return (evaluate_policy,)


@app.cell(hide_code=True)
def _(evaluate_policy, mo):
    policy_eval_button = mo.ui.button(label="Evaluate the simple_policy", value=[], kind="success", on_click= lambda value : evaluate_policy())

    mo.md(f'''
    Experiment with the simple_policy function definition above and the corresponding rewards from the policy **rollouts** (or **trajectories** resulting from following the policy.)

    {policy_eval_button}
    ''')
    return (policy_eval_button,)


@app.cell(hide_code=True)
def _(game, mo, policy_eval_button):
    mo.md(f"""
    {game.print_grid(policy_eval_button.value)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we see, following different policies results in the different rewards at the end of the  **episode** (note how we use the terms rollout, trajectory, episode interchangeably). The goal in RL is to find optimal policies resulting in largest possible reward for the agent.

    While learning the algorithms for finding such policies is outside the scope of this course, most popular methods rely on using some function approximators (like neural networks) to either learn the **value** function or the policy directly.

    Value functions are analogous to the optimal cost to the goal that we saw in the beginning of this notebook. Instead of giving the cost of the optimal path from each state, value functions give the total reward (or expected reward in the stochastic case) that the agent would gain by either following a specific policy (policy value function) or by following the optimal policy. To get more intution behind value functions we'll fill in the optimal value table analogous to the cost to the goal table.

    As you fill in this table refer to the slides and make sure you understand how these optimal values satisfy the Bellman Equation for the value function.
    """)
    return


@app.cell(hide_code=True)
def _(costs):
    rewards = {}
    for key in costs:
        rewards[key] = - costs[key]
    return (rewards,)


@app.cell(hide_code=True)
def _(game, mo, rewards):
    mo.md(f"""
    {game.print_values(rewards, game.value_input_grid, test=True)}
    """)
    return


@app.cell(hide_code=True)
def _(game, mo, rewards):
    values_button = mo.ui.button(label="Verify the Value Function", value="", kind="success", on_click= lambda value : game.print_values(rewards, game.value_input_grid))
    values_button
    return (values_button,)


@app.cell(hide_code=True)
def _(mo, values_button):
    mo.md(f"""
    {values_button.value}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In control problems, where the goal is to come up with an optimal policy, a more useful function is the **q-function**, which is analogous to the value function, but keeps track of the total expected reward based on not only the current state, but also the current action. Some of the RL algorithms use function approximators like neural networks to **learn** the value/q function (remember that in real scenario you don't have a map so cannot do the calculation above) by trying or observing many episodes or alternating policy improvements and q-function approximations under current policy.

    Once you know the optimal value/q functions figuring out a policy is straight-forward, as the agent takes the action with the maximum expected reward (given by value/q functions). This is referred to as 'acting greedily' with respect to it.

    Another common approach is to forego the intermediate step of learning the value function and instead learn directly the policy mapping $a(t) = \pi(s(t))$ from states to actions (this is particularly useful in the stochastic case when a neural network with a final softmax activation layer can give the probabilities of taking each action $a_i$ based on the current state $s(t)$,

    $$
    \pi(a_i(t) | s(t)).
    $$

    We finish with an example of one such method. Understanding the details of this and other methods is outside the scope of this course, but feel free to explore and play around with it below.
    """)
    return


@app.cell
def _(np):
    def policy_gradient(grid, terrain_costs, start, goal, episodes=1000, gamma=0.99, lr=0.001):
        rows, cols = len(grid), len(grid[0])
        actions = ['up', 'down', 'left', 'right']
        action_to_delta = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        n_actions = len(actions)

        # Initialize policy weights θ[state][action]
        theta = np.zeros((rows, cols, n_actions))

        def in_bounds(x, y):
            return 0 <= x < rows and 0 <= y < cols

        def get_reward(x, y):
            return -terrain_costs.get(grid[x][y], -float('inf'))

        def softmax(logits):
            exps = np.exp(logits - np.max(logits))  # for stability
            return exps / np.sum(exps)

        def choose_action(state):
            x, y = state
            probs = softmax(theta[x][y])
            action_index = np.random.choice(range(n_actions), p=probs)
            return actions[action_index], action_index, probs

        def step(state, action_name):
            dx, dy = action_to_delta[action_name]
            x, y = state
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                return (nx, ny), get_reward(nx, ny)
            return (x, y), get_reward(x, y)  # no-op with penalty

        # Training loop
        for episode in range(episodes):
            state = start
            trajectory = []
            for _ in range(100):  # max steps per episode
                action_name, action_idx, probs = choose_action(state)
                next_state, reward = step(state, action_name)
                trajectory.append((state, action_idx, reward))
                state = next_state
                if state == goal:
                    break

            # Compute returns and update θ
            G = 0
            for t in reversed(range(len(trajectory))):
                state, action_idx, reward = trajectory[t]
                G = reward + gamma * G
                x, y = state
                probs = softmax(theta[x][y])
                grad = -probs
                grad[action_idx] += 1  # ∇log π
                theta[x][y] += lr * G * grad  # policy gradient step

        # Generate best path using learned policy
        path = [start]
        state = start
        visited = set()
        for _ in range(100):
            x, y = state
            if state == goal or state in visited:
                break
            visited.add(state)
            probs = softmax(theta[x][y])
            action_name = actions[np.argmax(probs)]
            state, _ = step(state, action_name)
            path.append(state)

        total_cost = sum(get_reward(x, y) for x, y in path[1:-1])
        return total_cost, path

    return (policy_gradient,)


@app.cell
def _(game, policy_gradient):
    p_cost, p_trajectory = policy_gradient(game.grid, game.terrain_costs, game.start, game.goal, 10000)
    print(f"\nPolicy Gradient Cost: {p_cost}")
    print(f"Path: {p_trajectory}")
    return (p_trajectory,)


@app.cell(hide_code=True)
def _(game, mo, p_trajectory):
    mo.md(f"""
    {game.print_grid(p_trajectory)}
    """)
    return


if __name__ == "__main__":
    app.run()
