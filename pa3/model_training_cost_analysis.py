import argparse


def model_training_cost_analysis(model_config):
    # TODO: added your code here
    # You are free to add any helper functions and import any packages you see fit in this file
    raise NotImplementedError()

def get_optimal_N_D(training_budget):
    # TODO: added your code here
    # You are free to add any helper functions and import any packages you see fit in this file
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        num_parameters, num_flops, memory_cost = model_training_cost_analysis(args.model_config)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D = get_optimal_N_D(args.training_budget)
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    