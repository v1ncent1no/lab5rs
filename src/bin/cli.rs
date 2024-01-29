extern crate rand;

use rand::Rng;
use std::cmp::Ordering;
use std::f64;

const NEURON_COUNT: usize = 4;
const INPUT_COUNT: usize = 12;
const MAX_HEALTH: f64 = 100.0;
const REPRODUCTION_HEALTH_THRESHOLD: f64 = 80.0;
const PLANT_GENERATION_RATE: f64 = 0.1; // Chance to generate a new plant each cycle

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AgentType {
    Predator,
    Herbivore,
    Plant,
}

#[derive(Clone, Debug)]
struct Agent {
    agent_type: AgentType,
    health: f64,
    x: f64,
    y: f64,
    neural_network: NeuralNetwork,
}

#[derive(Clone, Debug)]
struct NeuralNetwork {
    weights: [[f64; INPUT_COUNT]; NEURON_COUNT],
}

impl NeuralNetwork {
    fn new() -> NeuralNetwork {
        let mut rng = rand::thread_rng();
        let weights =
            [[0.0; INPUT_COUNT]; NEURON_COUNT].map(|row| row.map(|_| rng.gen_range(-1.0..1.0)));
        NeuralNetwork { weights }
    }

    fn process_input(&self, input: [f64; INPUT_COUNT]) -> [f64; NEURON_COUNT] {
        let mut output = [0.0; NEURON_COUNT];
        for i in 0..NEURON_COUNT {
            output[i] = self.weights[i]
                .iter()
                .zip(input.iter())
                .map(|(w, inp)| w * inp)
                .sum();
        }
        output
    }
}

impl Agent {
    fn new(agent_type: AgentType, x: f64, y: f64) -> Agent {
        Agent {
            agent_type,
            health: 100.0,
            x,
            y,
            neural_network: NeuralNetwork::new(),
        }
    }

    fn update(&mut self, environment: &Environment) {
        // Get inputs from the environment
        let inputs = self.get_inputs(environment);

        // Process inputs through the neural network
        let outputs = self.neural_network.process_input(inputs);

        // Update position based on neural network output
        // Assuming that the first two neurons control vertical and horizontal movement
        self.x += outputs[0] - outputs[2]; // Right - Left
        self.y += outputs[1] - outputs[3]; // Down - Up

        // Update health (this is a placeholder, actual logic depends on the environment and interactions)
        self.health -= 1.0;

        // Ensure the agent stays within bounds
        self.x = self.x.clamp(0.0, environment.width);
        self.y = self.y.clamp(0.0, environment.height);

        log_agent_activity(self, "moved");

        if self.agent_type == AgentType::Predator || self.agent_type == AgentType::Herbivore {
            self.eat(&environment.agents);
        }

        if self.agent_type == AgentType::Predator {
            log_agent_activity(self, "tried to eat a herbivore");
        } else if self.agent_type == AgentType::Herbivore {
            log_agent_activity(self, "tried to eat a plant");
        }
    }

    fn eat(&mut self, agents: &[Agent]) {
        for agent in agents {
            if self.agent_type == AgentType::Predator && agent.agent_type == AgentType::Herbivore
                || self.agent_type == AgentType::Herbivore && agent.agent_type == AgentType::Plant
            {
                let distance = ((self.x - agent.x).powi(2) + (self.y - agent.y).powi(2)).sqrt();
                if distance < 1.0 {
                    self.health = (self.health + 20.0).min(MAX_HEALTH);
                    break;
                }
            }
        }
    }

    fn get_inputs(&self, environment: &Environment) -> [f64; INPUT_COUNT] {
        environment.get_inputs(self)
    }
}

#[derive(Debug, Clone)]
struct Environment {
    width: f64,
    height: f64,
    agents: Vec<Agent>,

    plant_generation_counter: f64,
}

impl Environment {
    fn new(width: f64, height: f64, agents: Vec<Agent>) -> Environment {
        Environment {
            width,
            height,
            agents,
            plant_generation_counter: 0.0,
        }
    }

    fn update(&mut self) {
        // Remove dead agents
        self.agents.retain(|agent| agent.health > 0.0);

        // Generate new plants
        self.plant_generation_counter += PLANT_GENERATION_RATE;
        if self.plant_generation_counter >= 1.0 {
            self.generate_plant();
            self.plant_generation_counter -= 1.0;
        }
    }

    fn generate_plant(&mut self) {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0.0..self.width);
        let y = rng.gen_range(0.0..self.height);
        self.agents.push(Agent::new(AgentType::Plant, x, y));
    }

    // ┌─────────┬─────────────────────────────────────────────┐
    // │ Input   │ Meaning                                     │
    // ├─────────┼─────────────────────────────────────────────┤
    // │ 0       │ Herbivore animal in the foreground          │
    // │ 1       │ Predator in the foreground                  │
    // │ 2       │ Plant in the foreground                     │
    // │ 3       │ Herbivore animal on the left                │
    // │ 4       │ Predator on the left                        │
    // │ 5       │ Plant on the left                           │
    // │ 6       │ Herbivore animal on the right               │
    // │ 7       │ Predator on the right                       │
    // │ 8       │ Plant on the right                          │
    // │ 9       │ Proximity of herbivore animal               │
    // │ 10      │ Proximity of predator                       │
    // │ 11      │ Proximity of plant                          │
    // └─────────┴─────────────────────────────────────────────┘
    fn get_inputs(&self, agent: &Agent) -> [f64; INPUT_COUNT] {
        let mut inputs = [0.0; INPUT_COUNT];

        for other in &self.agents {
            let dx = other.x - agent.x;
            let dy = other.y - agent.y;

            if dx == 0.0 && dy == 0.0 {
                continue;
            }

            let distance = (dx.powi(2) + dy.powi(2)).sqrt();

            let index = match other.agent_type {
                AgentType::Herbivore => {
                    if dx.abs() <= 1.0 && dy.abs() <= 1.0 {
                        0
                    }
                    // Foreground
                    else if dx < -1.0 {
                        3
                    }
                    // Left
                    else if dx > 1.0 {
                        6
                    }
                    // Right
                    else {
                        9
                    } // Proximity
                }
                AgentType::Predator => {
                    if dx.abs() <= 1.0 && dy.abs() <= 1.0 {
                        1
                    }
                    // Foreground
                    else if dx < -1.0 {
                        4
                    }
                    // Left
                    else if dx > 1.0 {
                        7
                    }
                    // Right
                    else {
                        10
                    } // Proximity
                }
                AgentType::Plant => {
                    if dx.abs() <= 1.0 && dy.abs() <= 1.0 {
                        2
                    }
                    // Foreground
                    else if dx < -1.0 {
                        5
                    }
                    // Left
                    else if dx > 1.0 {
                        8
                    }
                    // Right
                    else {
                        11
                    } // Proximity
                }
            };

            // Set input for the found agent
            inputs[index] = 1.0;
            // Proximity inputs are inversely proportional to distance
            if index >= 9 {
                inputs[index] = 1.0 / distance;
            }
        }

        inputs
    }
}

// fitness of an agent is his health
fn calculate_fitness(agent: &Agent) -> f64 {
    // Placeholder for fitness calculation
    agent.health // Example: using health as a fitness indicator
}

fn genetic_algorithm(agents: &mut Vec<Agent>) {
    let mut rng = rand::thread_rng();

    // Sort agents by fitness
    agents.sort_by(|a, b| {
        calculate_fitness(b)
            .partial_cmp(&calculate_fitness(a))
            .unwrap_or(Ordering::Equal)
    });

    let eligible_parents: Vec<_> = agents
        .iter()
        .filter(|a| a.health >= REPRODUCTION_HEALTH_THRESHOLD)
        .collect();
    if eligible_parents.is_empty() {
        return;
    }

    // Select the top N agents for breeding
    let top_n = agents.len() / 2; // Taking top half as an example
    let parents = &agents[..top_n];

    // Generate new agents through crossover and mutation
    let mut new_agents = Vec::new();
    while new_agents.len() < agents.len() - top_n {
        let parent1 = parents[rng.gen_range(0..top_n)].clone();
        let parent2 = parents[rng.gen_range(0..top_n)].clone();

        // Crossover - mix weights of two parents
        let mut child_weights = [[0.0; INPUT_COUNT]; NEURON_COUNT];
        for i in 0..NEURON_COUNT {
            for j in 0..INPUT_COUNT {
                child_weights[i][j] = if rng.gen_bool(0.5) {
                    parent1.neural_network.weights[i][j]
                } else {
                    parent2.neural_network.weights[i][j]
                };
            }
        }

        // Mutation - Adjust weights based on performance
        let fitness = calculate_fitness(&parent1);
        for i in 0..NEURON_COUNT {
            for j in 0..INPUT_COUNT {
                let adjustment = (1.0 - fitness) * rng.gen_range(-0.1..0.1); // Adjusting more for less fit agents
                child_weights[i][j] += adjustment;
                child_weights[i][j] = child_weights[i][j].clamp(-1.0, 1.0); // Keeping weights within bounds
            }
        }

        // Create a new agent with the child's weights
        let child_agent = Agent {
            agent_type: parent1.agent_type, // Assuming same type as parent
            health: (parent1.health + parent2.health) / 2.0,
            x: rng.gen_range(0.0..100.0), // Random initial position
            y: rng.gen_range(0.0..100.0),
            neural_network: NeuralNetwork {
                weights: child_weights,
            },
        };
        println!("Created a new agent: {:?}", child_agent);

        new_agents.push(child_agent);
    }

    let agents_len = agents.len() - 1;

    // Replace the least fit agents with new agents
    for i in 0..new_agents.len() {
        agents[agents_len - i] = new_agents[i].clone();
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let environment_width = 100.0;
    let environment_height = 100.0;
    let agent_count = 100; // Total number of agents (predators, herbivores, plants)
    let iterations = 1000; // Number of iterations for the simulation

    // Initialize agents with random types and positions
    let mut agents = Vec::new();
    for _ in 0..agent_count {
        let agent_type = match rng.gen_range(0..3) {
            0 => AgentType::Predator,
            1 => AgentType::Herbivore,
            _ => AgentType::Plant,
        };
        let x = rng.gen_range(0.0..environment_width);
        let y = rng.gen_range(0.0..environment_height);

        agents.push(Agent::new(agent_type, x, y));
    }

    let mut environment = Environment::new(environment_width, environment_height, agents);

    // Run the simulation
    for _ in 0..iterations {
        let mut agents_c = environment.agents.clone();

        for agent in agents_c.iter_mut() {
            agent.update(&mut environment);
        }

        // Apply the genetic algorithm
        genetic_algorithm(&mut environment.agents);

        // Additional logic for agent interactions (eating, reproducing, etc.) would go here
        environment.update();
    }

    // Print the neural network weights of the agents
    for (i, agent) in environment.agents.iter().enumerate() {
        println!(
            "Agent {}: Type: {:?}, Health: {:.2}, Position: ({:.2}, {:.2})",
            i, agent.agent_type, agent.health, agent.x, agent.y
        );
        println!("Neural Network Weights:");
        print_weights_table(&agent.neural_network.weights);
    }
}

fn print_weights_table(weights: &[[f64; INPUT_COUNT]; NEURON_COUNT]) {
    let directions = ["Up", "Down", "Left", "Right"];
    println!("┌────────┬────────────────────────────────────────────────────────────────────────────────────┐");
    println!("| Dir    | Weights                                                                            |");
    println!("├────────┼────────────────────────────────────────────────────────────────────────────────────┤");
    for (i, direction) in directions.iter().enumerate() {
        print!("| {:<6} |", direction);
        for weight in weights[i] {
            print!(" {:<5.2} ", weight);
        }
        println!("|");
    }
    println!("└────────┴────────────────────────────────────────────────────────────────────────────────────┘");
}

fn log_agent_activity(agent: &Agent, activity: &str) {
    println!(
        "Agent {:?} at ({:.2}, {:.2}) - {}",
        agent.agent_type, agent.x, agent.y, activity
    );
}
