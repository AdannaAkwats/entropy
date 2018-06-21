# Quantum Entropy
To run each python script, run:
`python file_name.py` on the  command line.

## Classical Shannon functions
**shannon.py**
- Contains Shannon definitions and its inequalities.
- Has function that generates random probability distributions.
- Contains definitions of all Shannon and non-Shannon inequalities.

**separate_probs.py**
Separates joint probability distributions into marginal distributions (and smaller joint distributions). Works up to 4 random variables.

## Von Neumann functions
**entropy.py**
Contains Von Neumann definitions and quantum inequalities.

**entangle.py**
- Contains definitions of Bell states and GHZ states
- Has functions that check for entanglement in quantum states.

**evolution.py**
- Functions of unitary evolution and unitary time evolution.
- Functions of quantum noisy channels: bit-flip, phase-flip, bit-phase-flip and depolarising channels.
- Has functions that check unitality, partial trace preservations and change of entropy of a quantum state when it passes through the quantum channels.

**non_shannon_quantum.py**
Contains definitions of the non-Shannon inequalities.

**partial_trace.py**
Contains function that computes partial trace for 2,3,4 and 5 qubit and qutrit systems.

**generate_random_quantum.py**
- Has functions that generate unitary, hermitian and density matrices (pure and mixed states).
- Different methods to generate mixed states.

## Testing
**run_classical_tests.py**
Runs Shannon and non-Shannon inequalities tests once. Uses *pytest*, so to run script, run `pytest --verbose run_classical_tests.py` on the command line.

**quantum_tests.py**
Has functions to used for testing all the Von Neumann and non-Shannon inequalities.

**extensive_testing.py**
Uses functions in *quantum_tests.py* to run tests all Von Neumann and non-Shannon inequalities. Running script runs goes through all inequalities and prints result in the command line. Uses similar syntax to *[pytest](https://docs.pytest.org/)*.

**print_tests.py**
Runs tests below and prints to command line.
*Tests*:
- Number of entangled states in mixed states generated
- All non-Shannon inequalities for the different generate functions, 100,000 times run.
- Non-Shannon inequality fir GHZ states.

## Other
**graphs.py**
  - Plots probability against average entropy difference (for 1000 states) for each quantum channel
  - Plots avg. difference with method of generation for Zhang-Yeung inequality

**utils.py**
Useful additional functions
