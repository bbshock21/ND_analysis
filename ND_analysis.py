import requests
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from collections import defaultdict

# CFBD API configuration
API_KEY = "API KEY HERE"  # Replace with your API key from https://collegefootballdata.com/key
BASE_URL = "https://api.collegefootballdata.com"

headers = {"Authorization": f"Bearer {API_KEY}"}

def get_team_games(team, year):
    """Get all games for a team in a given year"""
    url = f"{BASE_URL}/games"
    params = {"year": year, "team": team}
    response = requests.get(url, headers=headers, params=params)
    return response.json() if response.status_code == 200 else []

def get_fpi_ratings(year):
    """Get FPI ratings for all teams in a given year"""
    url = f"{BASE_URL}/ratings/fpi"
    params = {"year": year}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return {team['team']: team['fpi'] for team in data if team['fpi'] is not None}
    return {}

def get_conference_teams(year):
    """Get teams by conference for a given year"""
    url = f"{BASE_URL}/teams"
    params = {"year": year}
    response = requests.get(url, headers=headers, params=params)
    
    conf_teams = defaultdict(list)
    if response.status_code == 200:
        for team in response.json():
            if team['conference'] in ['ACC', 'Big Ten', 'Big 12', 'SEC', 'Pac-12']:
                conf_teams[team['conference']].append(team['school'])
    return conf_teams

def analyze_nd_schedule(start_year=2014, end_year=2024):
    """Analyze Notre Dame's ACC scheduling agreement"""
    
    results = {}
    
    for year in range(start_year, end_year + 1):
        # Skip 2020 COVID year
        if year == 2020:
            print(f"\nSkipping {year} (COVID year)")
            continue
            
        print(f"\nAnalyzing {year} season...")
        
        # Get Notre Dame's games
        nd_games = get_team_games("Notre Dame", year)
        
        # Get FPI ratings for the year
        fpi_ratings = get_fpi_ratings(year)
        
        # Get conference teams
        conf_teams = get_conference_teams(year)
        
        if not fpi_ratings:
            print(f"  No FPI data available for {year}")
            continue
        
        # Find ACC opponents Notre Dame played
        acc_opponents = []
        for game in nd_games:
            # Handle different possible key names in API response
            home = game.get('home_team') or game.get('homeTeam')
            away = game.get('away_team') or game.get('awayTeam')
            
            if not home or not away:
                print(f"  Warning: Could not parse game data: {game}")
                continue
                
            opponent = away if home == 'Notre Dame' else home
            
            # Check if opponent is in ACC
            if opponent in conf_teams['ACC']:
                acc_opponents.append(opponent)
        
        if not acc_opponents:
            print(f"  No ACC opponents found for {year}")
            continue
        
        # Calculate average FPI of ACC opponents
        acc_fpi_values = [fpi_ratings[team] for team in acc_opponents if team in fpi_ratings]
        
        if not acc_fpi_values:
            print(f"  No FPI ratings available for ACC opponents in {year}")
            continue
            
        avg_acc_fpi = np.mean(acc_fpi_values)
        num_acc_games = len(acc_opponents)
        
        print(f"  Notre Dame played {num_acc_games} ACC teams")
        print(f"  Average ACC opponent FPI: {avg_acc_fpi:.2f}")
        print(f"  ACC opponents: {', '.join(acc_opponents)}")
        
        # Calculate distributions for other conferences
        conf_distributions = {}
        
        for conf_name, teams in conf_teams.items():
            if conf_name == 'ACC':
                continue
            
            # Filter teams with FPI ratings
            teams_with_fpi = [t for t in teams if t in fpi_ratings]
            
            if len(teams_with_fpi) < num_acc_games:
                print(f"  Not enough {conf_name} teams with FPI data")
                continue
            
            # Calculate all possible combinations
            all_combinations = list(combinations(teams_with_fpi, num_acc_games))
            avg_fpis = []
            
            for combo in all_combinations:
                combo_fpi = np.mean([fpi_ratings[team] for team in combo])
                avg_fpis.append(combo_fpi)
            
            conf_distributions[conf_name] = avg_fpis
            
            print(f"  {conf_name}: mean={np.mean(avg_fpis):.2f}, "
                  f"median={np.median(avg_fpis):.2f}, "
                  f"std={np.std(avg_fpis):.2f}")
        
        results[year] = {
            'acc_opponents': acc_opponents,
            'acc_avg_fpi': avg_acc_fpi,
            'num_games': num_acc_games,
            'distributions': conf_distributions
        }
    
    return results

def visualize_results(results):
    """Create visualizations of the analysis"""
    
    years = sorted(results.keys())
    
    # Prepare data for plotting
    acc_avgs = [results[y]['acc_avg_fpi'] for y in years]
    
    conf_names = ['Big Ten', 'Big 12', 'SEC', 'Pac-12']
    conf_means = {conf: [] for conf in conf_names}
    
    for year in years:
        for conf in conf_names:
            if conf in results[year]['distributions']:
                conf_means[conf].append(np.mean(results[year]['distributions'][conf]))
            else:
                conf_means[conf].append(None)
    
    # Plot 1: Average FPI over time
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(years, acc_avgs, 'o-', linewidth=2, markersize=8, label='ACC (Actual)', color='darkblue')
    
    colors = {'Big Ten': 'red', 'Big 12': 'orange', 'SEC': 'purple', 'Pac-12': 'green'}
    for conf in conf_names:
        if any(v is not None for v in conf_means[conf]):
            plt.plot(years, conf_means[conf], 's--', linewidth=2, markersize=6, 
                    label=f'{conf} (Avg)', color=colors[conf], alpha=0.7)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average FPI Rating', fontsize=12)
    plt.title('Notre Dame: Average Opponent FPI by Conference', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Difficulty ranking
    plt.subplot(2, 1, 2)
    
    rankings = []
    for year in years:
        year_data = results[year]
        acc_fpi = year_data['acc_avg_fpi']
        
        rank = 1
        for conf in conf_names:
            if conf in year_data['distributions']:
                conf_mean = np.mean(year_data['distributions'][conf])
                if conf_mean > acc_fpi:
                    rank += 1
        rankings.append(rank)
    
    colors_rank = ['green' if r == 1 else 'yellow' if r == 2 else 'orange' if r == 3 else 'red' 
                   for r in rankings]
    plt.bar(years, rankings, color=colors_rank, edgecolor='black', linewidth=1.5)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Difficulty Rank (1=Hardest, 5=Easiest)', fontsize=12)
    plt.title('ACC Strength Ranking vs Other Power Conferences', fontsize=14, fontweight='bold')
    plt.ylim(0, 6)
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('nd_acc_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'nd_acc_analysis.png'")
    plt.show()
    
    # Create distribution plots for each season
    from scipy import stats
    
    num_years = len(years)
    cols = 3
    rows = (num_years + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if num_years > 1 else [axes]
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        year_data = results[year]
        acc_fpi = year_data['acc_avg_fpi']
        
        # Plot distribution for each conference
        for conf in conf_names:
            if conf in year_data['distributions']:
                distribution = year_data['distributions'][conf]
                
                # Fit gaussian to the distribution
                mu = np.mean(distribution)
                sigma = np.std(distribution)
                
                # Create x-axis range for smooth curve
                x_min = min(distribution) - sigma
                x_max = max(distribution) + sigma
                x = np.linspace(x_min, x_max, 200)
                
                # Calculate gaussian curve
                gaussian = stats.norm.pdf(x, mu, sigma)
                
                # Plot the curve
                ax.plot(x, gaussian, linewidth=2.5, label=conf, color=colors[conf], alpha=0.8)
        
        # Draw vertical line for ACC actual average
        ax.axvline(acc_fpi, color='darkblue', linewidth=3, linestyle='-', 
                   label='ACC (Actual)', alpha=0.9)
        
        ax.set_xlabel('Average FPI Rating', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'{year} Season', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for idx in range(num_years, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('nd_acc_distributions.png', dpi=300, bbox_inches='tight')
    print("Distribution plots saved as 'nd_acc_distributions.png'")
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Notre Dame ACC Scheduling Agreement Analysis")
    print("="*70)
    
    for year in years:
        year_data = results[year]
        acc_fpi = year_data['acc_avg_fpi']
        
        print(f"\n{year}:")
        print(f"  ACC opponents ({year_data['num_games']}): {', '.join(year_data['acc_opponents'])}")
        print(f"  ACC average FPI: {acc_fpi:.2f}")
        
        comparisons = []
        for conf in conf_names:
            if conf in year_data['distributions']:
                conf_mean = np.mean(year_data['distributions'][conf])
                diff = acc_fpi - conf_mean
                comparisons.append((conf, conf_mean, diff))
        
        comparisons.sort(key=lambda x: x[1], reverse=True)
        
        for conf, mean, diff in comparisons:
            if diff < 0:
                print(f"  {conf}: {mean:.2f} (ACC is {abs(diff):.2f} points EASIER)")
            else:
                print(f"  {conf}: {mean:.2f} (ACC is {diff:.2f} points HARDER)")

# Run the analysis
if __name__ == "__main__":
    print("Notre Dame ACC Scheduling Agreement Analysis")
    print("=" * 70)
    
    # Run analysis from 2014 to 2024 (excluding 2020)
    results = analyze_nd_schedule(2014, 2024)
    
    if results:
        visualize_results(results)
    else:
        print("No results to visualize. Check your API key and data availability.")
