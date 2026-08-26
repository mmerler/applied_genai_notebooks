# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Module 2: Practice 1 - Probability

    The goal of this practical is to get some hands-on practice with the concepts of probability and probability distributions, as well as, to establish some theoretical footing for the rest of the course. The math formulas you will see in this section can seem overwhelming at first, so do not worry if you don't follow everything. You should come back to these examples throughout the semester to deepen your understanding.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example 1: Probability Distributions and Entropy

    How can we analyze events that have random components to them? Let's extend the example of a hidden treasure in a box. This time, the color of the box hiding a treasure will be the random variable. Let's specify the total number of boxes of each color below and see how that affects the probability distribution of this random variable, assuming that the treasure is equally likely to be in any one of the boxes.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    num_colors = 4
    colors = ["black", "blue", "orange", "gold"]

    boxes_ui = mo.ui.array([mo.ui.text(value = f"{i:b}") for i in range(num_colors)])
    num_boxes_ui = mo.ui.array([mo.ui.number(label=colors[i], value = 2) for i in range(num_colors)])
    num_boxes_ui
    return colors, num_boxes_ui, num_colors


@app.cell(hide_code=True)
def _(colors, num_boxes_ui, num_colors, plt):
    import matplotlib.patches as patches

    # Define the box colors and labels
    box_colors = []
    num_boxes = 0
    for _i in range(num_colors):
        box_colors.extend([colors[_i]] * num_boxes_ui[_i].value)
        num_boxes += num_boxes_ui[_i].value

    box_labels = [f"Box {i+1}" for i in range(num_boxes)]

    # Define positions in two rows of 5 boxes
    box_positions = [(i % 5, -(i+1) // 5) for i in range(num_boxes)]  # Row layout

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 3))

    for _i, (_x, _y) in enumerate(box_positions):
        # Draw box
        rect = patches.Rectangle((_x, _y), 1, 1, facecolor=box_colors[_i], edgecolor='black')
        ax.add_patch(rect)

        # Add label in the center
        text_color = 'black' if box_colors[_i] == "gold" else 'white'
        ax.text(_x + 0.5, _y + 0.5, f"{box_labels[_i]}\n{box_colors[_i]}", ha='center', va='center', fontsize=10, color=text_color)

    # Formatting
    ax.set_xlim(0, 5)
    ax.set_ylim(-num_boxes//5, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f"{num_boxes} Boxes — One Contains a Treasure", fontsize=14)
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(colors, np, num_boxes_ui, num_colors, plt):
    color_counts = [num_boxes_ui[i].value for i in range(num_colors)]
    color_probs = color_counts/np.sum(color_counts)  # Normalize probabilities
    # Plot the distributions
    plt.figure(figsize=(8, 4))
    plt.bar(colors, color_probs, color=colors, width=0.3, alpha=0.8)
    plt.xlabel("Color")
    plt.ylabel("Probability")
    plt.title("Discrete Probability Distribution")
    plt.xticks(colors)
    plt.grid(True)
    plt.show()
    return (color_probs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Entropy: Measuring Information in a Random Variable

    One surprising, but extremely useful way to think about random variables and their probability distributions is through a concept from physics called **entropy**. In physics (statistical mechanics) entropy is related to the average amount of information needed to specify the microstate of a system. This has to do with energy levels, Boltzmann distributions and a LOT of math. If you are interested, take a look here: https://math.ucr.edu/home/baez/what_is_entropy/index.html.

    To understand entropy from information theory point of view, we turn to Claude Shannon and his famous work on "A Mathematical Theory of Communication" (https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf).

    Consider a random variable \( X \) defined by a **probability distribution** that represents probabilities of events, for example a treasure box having a particular color. A natural question arises:  
    **How much information do we gain when we observe a specific value sampled from this distribution?**

    Intuitively, we gain **more information** (alternatively **more surprise**) when we observe an event that is **unlikely** (i.e., has **low probability**) and **less information** (**less surprise**) when the event is **common** (i.e., has **high probability**).

    Furthermore, if we observe **two unrelated (independent) events**, the total information we gain should be the **sum of the information** from each individual event. Since the probabilities of independent events are multiplied, the function that turns probabilities to information should turn products of probabilities to sums of information. Turns out that mathematically the only function that has this property is the logarithm.

    These ideas lead us to a fundamental concept in Information Theory: **Entropy**.

    ---

    ### Definition of Entropy

    Formally, the **entropy** \( H(P) \) of a probability distribution \( P \) with possible outcomes \( x_1, x_2, \ldots, x_n \) and associated probabilities \( P(x_i) \) is defined as:

    \[
    H(P) = -\sum_{i} P(x_i) \log P(x_i).
    \]

    - The **logarithm** quantifies the amount of information gained from observing \( x_i \).
    - The **negative sign** ensures the entropy is non-negative.
    - The **sum** accounts for the expected information over all possible outcomes.

    Thus, entropy is a measure of the **average amount of information** we expect to gain from observing samples of a random variable. We can think of it as missing information or uncertainty associated with its value. 
    Entropy captures the **uncertainty** or **surprise** inherent in a probability distribution:

    - **Higher entropy** means more unpredictability (e.g., a fair coin toss).
    - **Lower entropy** means more certainty (e.g., a biased coin that almost always lands heads).

    Actually, Shannon in his original paper gave another definition of entropy as the **number of bits** required to optimally encode messages whose contents are chosen from a probability distribution, the idea being that we can encode common events using less bits. Bits are used as the measuring unit if the logarithm has base 2.  Changing to another base only results in a multiplicative factor in the definition of the entropy, so we can think of these differences as different units of measure for information, and in fact in machine learning applications a natural logarithm (base $e$) is typically used.

    Entropy is a foundational concept that will help us analyze and interpret **loss functions**, particularly in classification and generative models. 

    Note: we have to be careful with our intution since entropy is defined as the average information, so while low probability events have high entropy, they also contribute less to the sum. Experiment with the above distribution by changing the number of boxes of each color to get an intuition for high entropy and low entropy distributions.  What do you notice?
    """
    )
    return


@app.cell
def _(color_probs, np):
    def information(x):
        return np.log(x) if x != 0 else 0

    entropy = np.sum([-p*information(p) for p in color_probs])
    return (entropy,)


@app.cell(hide_code=True)
def _(color_probs, colors, entropy, num_boxes_ui, plt):
    # Plot the distributions
    plt.figure(figsize=(8, 4))
    plt.bar(colors, color_probs, color=colors, width=0.3, alpha=0.8)
    plt.xlabel("Color")
    plt.ylabel("Probability")
    plt.title(f"Discrete Probability Distribution.  Entropy: {entropy:.4f}")
    plt.xticks(colors)
    plt.grid(True)
    plt.show()
    num_boxes_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example 2: Cross-Entropy and KL Divergence

    Let's look at another example.  If you are driving on a highway at 67 miles per hour towards a police car with a radar gun, the radar will give a reading of your speed. However, there is a chance that the reading is not accurate due to various factors like interference, calibration issues, etc. This is where probability comes into play.

    Intuitively, there is certainly a larger probability that the speed will be shown as 66 mph than 50 mph. Is there a way for the radar manufacturer to describe the possible values the radar gun will give?

    One way, would be to define a probability distribution over the possible speeds. In this case the distribution is not discrete (we cannot list all of the possible speeds), but continuous. For simplicity we use the numbers for this specific example, but we can generalize this to any range of speeds and any distribution.

    Let's look at an example of two different radars, with the given probability distributions of speeds. Can you guess which distribution below has higher entropy?
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    from scipy.stats import norm
    from scipy.special import rel_entr

    # Define the range of speeds
    x = np.linspace(60, 74, 500)

    # Compute the PDF values
    _p = norm.pdf(x, loc=67, scale=0.5)
    _q = norm.pdf(x, loc=67, scale=2.0)

    # Normalize to make them proper discrete approximations of probability distributions
    _p /= np.sum(_p)
    _q /= np.sum(_q)

    # Plot the distributions
    plt.figure(figsize=(10, 5))
    plt.plot(x, _p, label=f"Radar A (μ=67, σ={0.5})", color='blue')
    plt.plot(x, _q, label=f"Radar B (μ=67, σ={2.0})", color='green')
    plt.fill_between(x, _p, alpha=0.3, color='blue')
    plt.fill_between(x, _q, alpha=0.3, color='green')
    plt.title("Continuous Radar Speed Distributions (Gaussian)")
    plt.xlabel("Measured Speed (mph)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    return norm, rel_entr, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In machine learning applications, the **primary probability distribution of interest** is the **data distribution** \( P(x) \), which represents how likely different data points are in the real world (we can think of $x$ as either the data points themselves or the possible outcomes in case of a supervised learning problem, technically $P(y | x)$).

    If we had access to the full distribution \( P(x) \), we could make **optimal predictions** of \( x \)..

    In practice, we **do not know** the true distribution $P(x)$.  
    Instead, we have:

    - A **set of samples** drawn from \( P(x) \)
    - An **approximate model distribution** \( Q(x) \), learned from the data

    We use \( Q(x) \) to make predictions and estimate probabilities.

    The **cross-entropy** between the true distribution \( P(x) \) and the model’s approximation \( Q(x) \) is given by:

    $$
    H[P, Q] = -\sum_x P(x) \log Q(x)
    $$

    Based on Shannon's definition this quantity can be interpreted as the **expected number of bits needed** (if we use based 2 logarithm) to encode samples from \( P \) **using a code optimized for** \( Q \).

    In other words, **cross-entropy measures how inefficient it is to approximate \( P \) using \( Q \)**. 

    - Lower cross-entropy means \( Q \) is a better approximation of \( P \)  
    - Minimizing the empirical cross-entropy is a **common loss function** in classification and generative modeling tasks

    $$
    -\frac{1}{N} \sum_{i=1}^{N} \log Q(x_i)
    $$

    Note: The term $P(x)$ is absorbed into the empirical averaging over the dataset.
    This is valid because the dataset is assumed to be an independent and identically distributed i.i.d. sample from the true $P(x)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Kullback-Leibler (KL) divergence

    We finish the discussion on entropy with another concept relating how close two probability distributions are, the **Kullback-Leibler (KL) divergence**.

    The **KL divergence** between two distributions \( P \) (the true distribution) and \( Q \) (the model’s predicted distribution) is defined as:

    \[
    D_{\text{KL}}(P \,\|\, Q) = \sum_x P(x) \log \left(\frac{P(x)}{Q(x)}\right)
    \]

    Since 

    $$
    \log \left(\frac{P(x)}{Q(x)}\right) = \log(P(x)) - \log(Q(x)),
    $$

    we have

    $$
    D_{\text{KL}}(P \,\|\, Q) =  \underbrace{H[P, Q]}_{cross-entropy} - \underbrace{H[P]}_{entropy}.
    $$

    this measures how many **extra** bits are needed to encode samples from \( P \) using a code optimized for \( Q \). A **lower KL divergence** indicates that \( Q \) is a better approximation of \( P \).

    KL divergence will be useful to compare two known or approximate distributions. You can experiment with changing the radar gaussian distributions below to see how the KL divergence changes. Note: with continuous distributions the definitions above involve integrals instead of sums, but intuitively are analogous.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_mu_1_ui = mo.ui.slider(label=r"$\mu_A$", value = 67, start=40, stop=90, step=1)
    slider_mu_2_ui = mo.ui.slider(label=r"$\mu_B$", value = 67, start=40, stop=90, step=1)

    slider_1_ui = mo.ui.slider(label=r"$\sigma_A$", value = 0.5, start=0.1, stop=3, step=0.1)
    slider_2_ui = mo.ui.slider(label=r"$\sigma_B$", value = 2.0, start=0.1, stop=3, step=0.1)

    slider_mu_1_ui, slider_1_ui, slider_mu_2_ui, slider_2_ui
    return slider_1_ui, slider_2_ui, slider_mu_1_ui, slider_mu_2_ui


@app.cell(hide_code=True)
def _(
    norm,
    np,
    plt,
    rel_entr,
    slider_1_ui,
    slider_2_ui,
    slider_mu_1_ui,
    slider_mu_2_ui,
    x,
):
    mu_p, sigma_p = slider_mu_1_ui.value, slider_1_ui.value  # Radar Gun A: more precise
    mu_q, sigma_q = slider_mu_2_ui.value, slider_2_ui.value  # Radar Gun B: less precise

    p = norm.pdf(x, loc=mu_p, scale=sigma_p)
    q = norm.pdf(x, loc=mu_q, scale=sigma_q)

    # Normalize to make them proper discrete approximations of probability distributions
    p /= np.sum(p)
    q /= np.sum(q)


    kl_pq = np.sum(rel_entr(p, q))  # KL(P || Q)
    kl_qp = np.sum(rel_entr(q, p))  # KL(Q || P)

    # Plot the distributions
    plt.figure(figsize=(10, 5))
    plt.plot(x, p, label=f"Radar A (μ=67, σ={sigma_p})", color='blue')
    plt.plot(x, q, label=f"Radar B (μ=67, σ={sigma_q})", color='green')
    plt.fill_between(x, p, alpha=0.3, color='blue')
    plt.fill_between(x, q, alpha=0.3, color='green')
    plt.title(f"Continuous Radar Speed Distributions (Gaussian). KL(P || Q): {kl_pq:.4f}")
    plt.xlabel("Measured Speed (mph)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example 3: Bayes Theorem (Optional)

    To get comfortable with handling probability expressions, we will play a treasure hunt game on a 5x5 grid.
    The rules are simple:

    - There is a treasure hidden somewhere on the grid (💎).
    - You have to guess where it is in a minimum number of tries.
    - Every time you guess I give you a distance clue: the manhattan distance to the treasure.
    - To make this a bit more challenging:
        - 80% of the time the clue will be correct.
        - 10% of the time it will be off by 1.
        - 10% of the time it will be off by -1.
    - Click on different cells below to start the game!
    - Click on 'Simulate 1 Turn' to choose cell based on the highest probability and update the probability table based on the latest distance clue.
    """
    )
    return


@app.cell(hide_code=True)
def _(buttons, field, game, mo, prob):
    mo.md(fr'''

    {mo.md('YOU WIN' if game.treasure_location == game.current_guess else f"Distance clue:  {game.distance_clue}").callout(kind="success")}

    {field}   

    {buttons}

    Probability Table (only updated when Simulating Turns):

    {prob} 


    Guesses: {game.seeker_guesses}
    ''')
    # Treasure Location: {game.treasure_location}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Explanation

    How does the 'Simulate 1 Turn' work to find the location in only a few tries?

    Can we use probability theory to make this efficient?

    Note that I gave you the probabilities of what the distance clue should be, given the treasure location. We want to *reverse* that probability and obtain the probability of treasure location given the distance clue. This is a perfect application of Bayes Theorem!

    Notation:

    Let's use $T_{ij}$ as the random variable that describes the existence of treasure in cell $(i,j)$.

    We will say that $T_{ij} = 🌱$ if there is no treasure in that cell and $T_{ij} = 💎$ if there is.

    After each turn we get a distance clue. Let $d_t$ represent the distance clue we get at turn $t$. 

    So $P(T_{ij} = 💎 | d_1)$ represents our belief after 1 turn that $T_{ij}$ contains treasure.

    $P(T_{ij} = 💎  | d_t, ..., d_2, d_1)$ represents our belief after $t$ turns ($t$ clues) that the cell $(i,j)$ contains treasure.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    number_array = mo.ui.array([mo.ui.number(value = None) for i in range(5)])
    mo.md(
        f"""
        ---
        ## Derivation

        Now let's add the information we have

        Let's try to calculate the different probabilities involved. 

        Say the treasure is in location $(1,2)$ and we guessed $(1,4)$ on our first turn. 
        What is the probability that the distance clue will be given as 0 (check the rules of the game in the beginning)?

        $P(d_1 = 0 | T_{{ij}} = 💎) =$ {number_array[0]}
        """
    )
    return (number_array,)


@app.cell(hide_code=True)
def _(mo, number_array):
    mo.md(
        f"""
    $P(d_1 = 0 | T_{{ij}} = 💎) = {number_array[0].value}$
    {'✅' if number_array[0].value == 0 
    else '❌'}

    {'Correct! The probability that the distance if off by 2 units is actually 0.' if number_array[0].value == 0 else 'Note that the real distance between those locations is 2, so what is the probability that the clue is off by 2 units?'}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, number_array):
    mo.md(
        f"""
    What about the probability that the distance clue will be given as 1?

    $P(d_1 = 1 | T_{{ij}} = 💎) =$ {number_array[1]} &nbsp;&nbsp; $P(d_1 = -1 | T_{{ij}} = 💎) =$ {number_array[2]}

    $P(d_1 = 2 | T_{{ij}} = 💎) =$ {number_array[3]} &nbsp;&nbsp; $P(d_1 = -2 | T_{{ij}} = 💎) =$ {number_array[4]}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can now write a more general formula of the probability of the distance clue being given as $d$ if  the treasure is in cell (i,j) and our guess was cell (k,l):

    $$
    P(d_t = d | T_{ij} = 💎) = 
    \begin{cases}
    0.8 & \text{if distance((k,l), (i,j)) = d} \\
    0.2 & \text{if distance((k,l), (i,j)) = d $\pm 1$} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    And using Bayes Formula:

        $$P(T_{ij} = 💎 | d_t = d, d_{t-1} ..., d_1) = \frac{\overbrace{P(d_t = d| T_{ij} = 💎, d_{t-1}, ..., d_1)}^{likelihood} \overbrace{P(T_{ij} = 💎, d_{t-1}, ..., d_1)}^{prior}}{P(d_1, d_2, ..., d_{t})}$$

    Note how the prior gets updated after each new clue by being multiplied by the likelihood. We don't have to worry about the denominator since it is the same for all of the cells, so as long as we normalize the probabilities at the end we are good to go.

    ----
    ## Implementation

    The part of implementation that we want to study deals with the logic of how the algorithm decides where to search next based on the probability table update that uses the previous guess and the distance clue received after the previous guess. For each cell it calculates the distance to the  guess location (i.e. the correct distance if that cell had the treasure) and compares it with the distance clue received. This is the 'update_probabilities' function. It is just the implementation of the above formula.
    """
    )
    return


@app.cell
def _(np):
    def update_probabilities(probability_grid, guess, distance_clue):
        """ Update the probability grid using Bayesian inference """
        new_grid = np.zeros_like(probability_grid)
        total_prob = 0

        for i in range(probability_grid.shape[0]):
            for j in range(probability_grid.shape[1]):
                estimated_distance = abs(i - guess[0]) + abs(j - guess[1])
                if estimated_distance == distance_clue: # correct clue
                    likelihood = 0.6
                elif ((estimated_distance - distance_clue) == 1 or 
                     (estimated_distance - distance_clue) == -1): # off by +1 or -1
                    likelihood = 0.2
                else:
                    likelihood = 0
                new_grid[i, j] = probability_grid[i, j] * likelihood
                total_prob += new_grid[i, j]

        if total_prob > 0:
            new_grid /= total_prob
        return new_grid
    return (update_probabilities,)


@app.cell(hide_code=True)
def _(np, update_probabilities):
    import matplotlib.pyplot as plt

    class BayesianTreasureHunt:
        def __init__(self, grid_size=5):
            self.grid_size = grid_size
            self.reset_game()

        def reset_game(self):
            self.treasure_location = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.probability_grid = np.full((self.grid_size, self.grid_size), 1 / (self.grid_size**2))
            self.seeker_guesses = []
            self.current_guess = None
            self.distance_clue = None
            self.iterations = 0

        def provide_clue(self, guess):
            """ Provides a noisy clue based on a distribution around the true distance with 20% incorrect information """
            true_distance = abs(guess[0] - self.treasure_location[0]) + abs(guess[1] - self.treasure_location[1])
            noise = np.random.choice([0, -1, 1], p=[0.6, 0.2, 0.2]) 
            return max(0, true_distance + noise)

        def play_round_manual(self, guess):
            self.current_guess = guess
            self.seeker_guesses.append(self.current_guess)
            distance = self.provide_clue(self.current_guess)
            self.iterations += 1
            self.distance_clue = distance
            return distance

        def play_round(self):
            """ Plays one round of the game """
            self.current_guess = tuple(np.unravel_index(np.argmax(self.probability_grid, axis=None), self.probability_grid.shape))
            self.seeker_guesses.append(self.current_guess)
            if (self.current_guess == self.treasure_location):
                return 0
            else:
                self.probability_grid[self.current_guess] = 0
            distance_clue = self.provide_clue(self.current_guess)
            self.probability_grid = update_probabilities(self.probability_grid, self.current_guess, distance_clue)
            self.iterations += 1
            self.distance_clue = distance_clue
            return distance_clue

        def get_game_state(self):
            return {
                "Treasure Location": self.treasure_location[::-1],
                "Seeker Guesses": np.flip(self.seeker_guesses, axis=1),
                "Iterations": self.iterations,
                "Current Guess": self.current_guess[::-1]
            }

    # Run an interactive session
    game = BayesianTreasureHunt(grid_size=5)
    return game, plt


@app.cell(hide_code=True)
def _(game, mo):
    # Grid Button Code

    ui_grid = mo.ui.dictionary({
        f"{i,j}": mo.ui.button(label=f"{i,j}", value=(i,j),
            on_click= lambda value : game.play_round_manual(value)) 
                for i in range(game.grid_size) for j in range(game.grid_size)
    })
    button = mo.ui.button(label="Simulate 1 Turn", value="", kind="success", on_click= lambda value : game.play_round())
    reset_button = mo.ui.button(value=None, label="Reset Game", on_click= lambda value : game.reset_game())
    return button, reset_button, ui_grid


@app.cell(hide_code=True)
def _(button, game, reset_button, ui_grid):
    # Grid Graphics Code

    field = "|"
    for j in range(game.grid_size):
        field = field + "         |"
    field = field + "\n|"
    for j in range(game.grid_size):
        field = field + "-------- |"
    field = field + "\n"
    for i in range(game.grid_size):
        field = field + "| "
        for j in range(game.grid_size):
            field = field + ("💎" if ((i,j) in game.seeker_guesses) and ((i,j) == game.treasure_location)
                             else "🌱" if (i,j) in game.seeker_guesses else f"{ui_grid[f'({i}, {j})']}")
            field = field + " |"
        field = field + "\n"

    prob = "|"
    for j in range(game.grid_size):
        prob = prob + "         | "
    prob = prob + "\n| "
    for j in range(game.grid_size):
        prob = prob + "-------- | "
    prob = prob + "\n| "
    for i in range(game.grid_size):
        for j in range(game.grid_size):
            prob = prob + str(round(game.probability_grid[i,j], 2)) + " |"
        prob = prob + "\n|"

    buttons = f"{reset_button}" + " " + (f"{button}" if game.treasure_location != game.current_guess else "")
    return buttons, field, prob


if __name__ == "__main__":
    app.run()
