import matplotlib.pyplot as plt

x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y = [1.8324, 0.2450, 0.2134, 0.1763, 0.1683, 0.0016, 0.1370, 0.1835, 0.0008, 0.0398, 0.0738]

plt.plot(x, y, marker='o', linestyle='-', color='red')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("RNN Error V/S Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("Plot generation complete. The plot window should now be displayed.")

