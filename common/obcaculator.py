import re
from collections import defaultdict
import sys # Import sys to allow exiting

# Atomic weights of common elements (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085,
    'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
    'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38,
    # --- Added Elements ---
    'B': 10.81,   # Boron
    'F': 18.998,  # Fluorine
    'Ti': 47.867, # Titanium
    'Mn': 54.938, # Manganese
    'Br': 79.904, # Bromine
    # More elements can be added as needed
}


def parse_formula(formula: str) -> dict:
    """
    Parses a chemical formula string and returns a dictionary containing elements and their counts.
    Catches and returns ValueError to be handled at a higher level.
    """
    pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    matches = pattern.findall(formula)
    if not matches and formula.strip(): # Handle non-empty but unmatchable cases
        raise ValueError(f"Could not parse formula '{formula}'. Please ensure the format is correct (e.g., C6H14O6, KNO3).")
    if not formula.strip(): # Handle empty strings
        raise ValueError("Formula cannot be empty.")

    atom_counts = defaultdict(int)
    parsed_elements_string = "" # Used to verify if the parsing covered the entire string
    for element, count_str in matches:
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Error: Unknown element '{element}' in formula '{formula}'. Please add it to the ATOMIC_WEIGHTS dictionary.")
        count = int(count_str) if count_str else 1
        atom_counts[element] += count
        parsed_elements_string += element + count_str

    # Check for unparsed parts (simple integrity check)
    if parsed_elements_string != formula.replace(" ", ""): # Ignore spaces in comparison
         # More complex checks might be needed, but this catches some basic errors
        pass # Allow partial matches, but a warning could be added for stricter checks

    if not atom_counts: # If the dictionary is still empty after the loop
        raise ValueError(f"Could not extract any elements from '{formula}'.")

    return dict(atom_counts)

def calculate_mw(atom_counts: dict) -> float:
    """Calculates the molecular weight based on atom counts."""
    mw = 0.0
    for element, count in atom_counts.items():
        mw += ATOMIC_WEIGHTS[element] * count
    return mw

def calculate_ob_percent(atom_counts: dict, mw: float) -> float:
    """
    Calculates the oxygen balance (OB%) of a compound.
    Based on the assumption of complete oxidation: C -> CO2, H -> H2O.
    """
    if mw == 0:
        return 0.0 # Avoid division by zero

    x = atom_counts.get('C', 0)
    y = atom_counts.get('H', 0)
    z = atom_counts.get('O', 0)

    # Apply the oxygen balance formula: OB% = (z - 2*x - y/2) * 16 / mw * 100
    ob_percent = (z - 2 * x - y / 2) * 15.999 / mw * 100
    return ob_percent

def get_user_input():
    """Interactively gets component chemical formulas and proportion information from the user."""
    # component_data_input will store (formula, proportion) tuples
    component_data_input = []

    while True:
        try:
            num_components = int(input("Please enter the number of components in the mixture: "))
            if num_components > 0:
                break
            else:
                print("Number of components must be greater than 0.")
        except ValueError:
            print("Invalid input, please enter an integer.")

    print("-" * 30)
    formulas_seen = set() # Used to check for duplicate formula inputs

    for i in range(num_components):
        print(f"--- Enter Component #{i+1} ---")
        while True:
            formula = input(f"Please enter the chemical formula for component #{i+1} (e.g., C6H14O6, KNO3): ").strip()
            if not formula:
                print("Chemical formula cannot be empty.")
                continue
            # Check if the formula has already been entered
            if formula in formulas_seen:
                 print(f"Warning: The formula '{formula}' has already been entered.")
                 # You can choose to prevent duplicate entries or allow them with a warning.
                 # Here, we allow duplicates, but they will be treated as different parts of the same substance in later calculations.
                 # If you want to prevent duplicates, you can uncomment the lines below and add 'continue'.
                 # print("Please enter a unique chemical formula, or combine their proportions before entering.")
                 # continue

            try:
                # Try to parse here to catch errors early
                parse_formula(formula)
                # formulas_seen.add(formula) # If duplicates are allowed, it can be added here
                break # Formula is valid, break the loop
            except ValueError as e:
                print(f"Formula Error: {e}")
                print("Please re-enter.")

        while True:
            proportion_str = input(f"Please enter the mass proportion for formula '{formula}' (e.g., 70 or 30): ").strip()
            try:
                proportion = float(proportion_str)
                if proportion >= 0:
                    component_data_input.append((formula, proportion))
                    formulas_seen.add(formula) # Add to the set of seen formulas
                    break
                else:
                    print("Mass proportion cannot be negative.")
            except ValueError:
                print("Invalid input, please enter a number.")
        # print("-" * 10) # Small separator after each component input (optional)

    return component_data_input



def calculate_and_display_results(component_data_input: list):
    """
    Performs calculations and displays the results.
    Now receives a list of (formula, proportion) tuples.
    """
    print("\n" + "="*15 + " Calculation Results " + "="*15)

    # --- Individual Component Calculations ---
    # component_details stores the calculated details for each entry
    component_details = []
    print("--- Individual Component Calculation Results ---")
    calculation_successful = True
    total_proportion = 0.0

    # Used to aggregate calculation results for the same chemical formula
    aggregated_data = {} # key: formula, value: {'mw': mw, 'ob_percent': ob, 'total_proportion': prop_sum}

    for formula, proportion in component_data_input:
        try:
            # If this formula has been calculated before, reuse the result
            if formula in aggregated_data:
                mw = aggregated_data[formula]['mw']
                ob_percent = aggregated_data[formula]['ob_percent']
                aggregated_data[formula]['total_proportion'] += proportion
            else:
                # Otherwise, perform the calculation
                atom_counts = parse_formula(formula)
                mw = calculate_mw(atom_counts)
                ob_percent = calculate_ob_percent(atom_counts, mw)
                # Store the calculation results for reuse and final display
                aggregated_data[formula] = {
                    'mw': mw,
                    'ob_percent': ob_percent,
                    'total_proportion': proportion
                }
                # When encountering this formula for the first time, print its basic information
                print(f"Component (Formula): {formula}")
                print(f"  Molecular Weight (Mw): {mw:.3f} g/mol")
                print(f"  Oxygen Balance (OB%): {ob_percent:+.2f}%")

            # Record the details of each input entry, including its original proportion and calculated properties
            component_details.append({
                'formula': formula,
                'proportion': proportion,
                'mw': mw,
                'ob_percent': ob_percent
            })
            total_proportion += proportion

        except ValueError as e:
            print(f"Error processing formula '{formula}': {e}")
            calculation_successful = False
            # Even if one component fails, try to continue processing others

    print("-" * 30)

    if not calculation_successful:
        print("Could not complete mixture calculation due to errors in one or more components.")
        return

    if total_proportion <= 0:
        print("Warning: The total proportion of all components is 0 or negative. Cannot calculate mixture percentages and oxygen balance.")
        return

    # --- Mixture Calculation ---
    print("--- Mixture Calculation Results ---")
    mixture_ob_percent = 0.0
    print("Actual Mass Percentage:")

    # Display percentages aggregated by chemical formula
    for formula, data in aggregated_data.items():
        mass_fraction = data['total_proportion'] / total_proportion
        actual_percentage = mass_fraction * 100
        print(f"  - {formula}: {actual_percentage:.2f}%")
        # Use the total mass fraction of each chemical and its OB% to calculate the mixture OB%
        mixture_ob_percent += mass_fraction * data['ob_percent']

    # # Alternatively, if you want to display each input entry and its percentage (even if formulas are repeated)
    # print("Actual Mass Percentage by Input Entry:")
    # mixture_ob_percent_alt = 0.0
    # for detail in component_details:
    #     mass_fraction = detail['proportion'] / total_proportion
    #     actual_percentage = mass_fraction * 100
    #     print(f"  - {detail['formula']} (input proportion {detail['proportion']}): {actual_percentage:.2f}%")
    #     mixture_ob_percent_alt += mass_fraction * detail['ob_percent']
    # # Note: mixture_ob_percent and mixture_ob_percent_alt should yield the same result.

    print(f"\nTotal Mixture Oxygen Balance (OB%): {mixture_ob_percent:+.2f}%")
    print("-" * 30)

    # ... (The following hints and safety warnings remain unchanged) ...

    if mixture_ob_percent > 0:
        print("Hint: The mixture has a positive oxygen balance, meaning there is excess oxygen available to oxidize other materials or produce oxygen-rich gas.")
    elif mixture_ob_percent < 0:
        print("Hint: The mixture has a negative oxygen balance, meaning there is insufficient oxygen to completely oxidize all combustible materials, potentially producing incomplete combustion products like CO, H₂, or soot.")
    else:
        print("Hint: The mixture has a zero oxygen balance (or close to it), meaning theoretically, the amount of oxygen is just sufficient to completely oxidize the combustible materials.")

    print("\nImportant Safety Warning:")
    print("Mixtures involving oxidizers and fuels can be explosive or flammable.")
    print("These calculations are theoretical values only, and actual behavior can be affected by many factors.")
    print("Do not attempt to prepare or test these mixtures without professional knowledge, proper equipment, and safety measures.")
    print("="*40)




# --- Main Program Loop ---
if __name__ == "__main__":
    print("Welcome to the Mixture Oxygen Balance Calculator!")
    while True:
        try:
            # Call the get_user_input function
            component_data_list = get_user_input()
            # Call the calculate_and_display_results function
            calculate_and_display_results(component_data_list)
        except ValueError as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your input.")
        except Exception as e: # Catch other unexpected errors
             print(f"\nAn unexpected error occurred: {e}")

        # ... (The part asking whether to continue remains unchanged) ...
        while True:
            another = input("\nDo you want to perform another calculation? (y/n): ").lower().strip()
            if another == 'y' or another == 'yes':
                print("\n" + "="*50 + "\n") # Clearly start the next calculation
                break
            elif another == 'n' or another == 'no':
                print("Thank you for using the program. Exiting.")
                sys.exit() # Exit the program
            else:
                print("Please enter 'y' or 'n'.")
