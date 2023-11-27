import matplotlib.pyplot as plt
def parse_input():
    x = []
    y = []
    
    try:
        while True:
            line = input("Enter space-separated values (or press Enter to finish): ")
            if not line.strip():
                break  # Exit the loop if an empty line is entered
            
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
        
    except KeyboardInterrupt:
        print("\nInput interrupted. Exiting.")

    return x, y

if __name__ == "__main__":
    try:
        x_values, y_values = parse_input()
        
        print("x:", x_values)
        print("y:", y_values)

        plt.plot(x_values, y_values)
        plt.show()
            
    except Exception as e:
        print(f"Error: {e}")