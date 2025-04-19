#!/usr/bin/env python3
"""
Vehicle Routing Problem solver using Ant Colony Optimization with 2-Opt local search.

This module provides functions for parsing VRP XML instances and solving them via ACO,
with route improvement via 2-Opt after each ant's tour construction.
It can be executed as a script ("python solution.py ...") or its functions can be imported in tests.
"""
import xml.etree.ElementTree as ET
from collections import namedtuple
import math
import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

# Node data structure
Node = namedtuple("Node", ["id", "x", "y", "demand", "is_depot"])

def parse_vrp(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Parse vehicle capacity
    capacity = None
    vp = root.find('.//fleet/vehicle_profile')
    if vp is not None:
        cap_el = vp.find('capacity')
        if cap_el is not None and cap_el.text:
            capacity = float(cap_el.text)
    if capacity is None:
        raise ValueError(f"Could not find vehicle capacity in {xml_file}")
    # Parse demands
    demands = {}
    for req in root.findall('.//requests/request'):
        node_id = int(req.get('node'))
        qty_el = req.find('quantity')
        if qty_el is not None and qty_el.text:
            demands[node_id] = float(qty_el.text)
    # Parse nodes
    nodes = []
    for n in root.findall('.//network/nodes/node'):
        nid = int(n.get('id'))
        t = n.get('type')
        is_depot = (t == '0' or t.lower() == 'depot')
        cx = n.find('cx')
        cy = n.find('cy')
        if cx is None or cy is None:
            raise ValueError(f"Missing coordinates for node {nid} in {xml_file}")
        x = float(cx.text)
        y = float(cy.text)
        demand = 0.0 if is_depot else demands.get(nid, 0.0)
        nodes.append(Node(nid, x, y, demand, is_depot))
    if not nodes:
        raise ValueError(f"No nodes parsed from {xml_file}")
    nodes.sort(key=lambda nd: nd.id)
    return nodes, capacity

def compute_distance_matrix(nodes):
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx, dy = nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y
            dist[i, j] = math.hypot(dx, dy)
    return dist

# Sourced from tutorial 06
def initialize_pheromone(n):
    return 0.01 * np.ones(shape=(n, n))

# --- Local search: 2-Opt ---
def route_cost(route, dist):
    """Compute total distance of a single route (including return to depot)."""
    return sum(dist[i, j] for i, j in zip(route[:-1], route[1:]))

def two_opt(route, dist):
    """
    Improve a single route using the 2-Opt heuristic.
    """
    best = route
    best_cost = route_cost(best, dist)
    improved = True
    while improved:
        improved = False
        n = len(best)
        # skip depot at positions 0 and n-1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route, dist)
                if new_cost < best_cost:
                    best, best_cost = new_route, new_cost
                    improved = True
                    break  # improvement found, restart
            if improved:
                break
    return best

# Sourced from tutorial 06
# Slightly modified to avoid arrays missmatch
def update_pheromone(pheromone, solutions, costs, rho, Q):
    # Evaporation
    pheromone *= (1 - rho)
    # Deposition
    for sol, cost in zip(solutions, costs):
        deposit = Q / cost
        for route in sol:
            for i, j in zip(route[:-1], route[1:]):
                pheromone[i, j] += deposit
                pheromone[j, i] += deposit
    return pheromone

# --- Construct initial solutions ---
def build_solution(nodes, dist, pheromone, capacity, alpha, beta):
    depot = next((i for i, nd in enumerate(nodes) if nd.is_depot), None)
    if depot is None:
        raise ValueError("No depot node found in instance")
    to_serve = {i for i, nd in enumerate(nodes) if not nd.is_depot}
    solution = []
    while to_serve:
        route = [depot]
        load = 0.0
        curr = depot
        while True:
            feasible = [i for i in to_serve if load + nodes[i].demand <= capacity]
            if not feasible:
                break
            probs = np.array([
                pheromone[curr, j]**alpha * (1.0/(dist[curr, j] + 1e-6))**beta
                for j in feasible
            ])
            probs /= probs.sum()
            next_node = np.random.choice(feasible, p=probs)
            route.append(next_node)
            load += nodes[next_node].demand
            to_serve.remove(next_node)
            curr = next_node
        route.append(depot)
        solution.append(route)
    return solution

# --- Cost evaluation ---
def cost_of_solution(solution, dist):
    total_distance = sum(dist[i, j] for r in solution for i, j in zip(r[:-1], r[1:]))
    vehicles_used = len(solution)
    return total_distance, total_distance, vehicles_used

# --- Main ACO loop with 2-Opt refinement ---
def ant_colony_vrp(nodes, dist, capacity,
                   num_ants=20, max_iters=100,
                   alpha=1.0, beta=4.0, rho=0.2, Q=1):
    pheromone = initialize_pheromone(len(nodes))
    best_solution = None
    best_cost = float('inf')
    history = []

    for iteration in range(max_iters):
        solutions, costs = [], []
        for _ in range(num_ants):
            sol = build_solution(nodes, dist, pheromone, capacity, alpha, beta)
            # Apply 2-Opt to each route
            sol = [two_opt(route, dist) for route in sol]
            cost, dist_val, veh = cost_of_solution(sol, dist)
            solutions.append(sol)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_solution = (sol, dist_val, veh)
        history.append(best_cost)
        pheromone = update_pheromone(pheromone, solutions, costs, rho, Q)
    return best_solution, history

# --- Instance solver and script entrypoint ---
def solve_instance(xml_file, **params):
    nodes, capacity = parse_vrp(xml_file)
    dist = compute_distance_matrix(nodes)
    (solution, total_dist, vehicles), history = ant_colony_vrp(
        nodes, dist, capacity, **params)
    return {
        'routes': solution,
        'distance': total_dist,
        'vehicles': vehicles,
        'history': history
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Solve VRP using ACO with 2-Opt")
    parser.add_argument('paths', nargs='+', help='XML files or directories containing XMLs')
    return parser.parse_args()


def plot_convergence(inst, result, path):
    plt.figure(figsize=(10, 6))
    plt.plot(result['history'], linestyle='-', linewidth=1.5, label='Best Cost')
    iterations = list(range(len(result['history'])))
    plt.scatter(iterations, result['history'], s=30, zorder=5)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    title = f"Convergence: {os.path.basename(inst)}"
    plt.title(title, pad=15)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = f"{os.path.splitext(os.path.basename(inst))[0]}.png"
    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Convergence plot saved to {plot_path}")


def collect_xml_instances(paths):
    instances = []
    for p in paths:
        if os.path.isdir(p):
            instances.extend(sorted(glob.glob(os.path.join(p, '*.xml'))))
        elif os.path.isfile(p) and p.lower().endswith('.xml'):
            instances.append(p)
    if not instances:
        print("No XML instances found in the provided paths.")
        return None
    return instances


def main():
    args = parse_args()
    instances = collect_xml_instances(args.paths)
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    for inst in instances:
        print(f"Solving {inst}")
        try:
            result = solve_instance(inst)
        except Exception as e:
            print(f"Error solving {inst}: {e}")
            continue
        print(f"Vehicles: {result['vehicles']}, Distance: {result['distance']:.2f}")
        for route in result['routes']:
            print(' -> '.join(map(str, route)))
        plot_convergence(inst, result, plots_dir)

if __name__ == '__main__':
    main()
