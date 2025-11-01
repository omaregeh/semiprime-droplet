> **Note:** This is an experimental visual exploration in number theory.

# A Surprising Droplet Shape Hidden in Semiprimes

When plotting the products of two prime numbers (called **semiprimes**), an unexpected shape appears — a smooth, tilted **droplet** with clear internal stripes. This project explores that pattern and helps you visualize how it forms.

![Example droplet](example_output.png)

---

## What This Project Does

This code generates all semiprimes up to a selected limit and plots them using a transformation that produces a distinctive **droplet-shaped pattern**. It also shows why the shape forms by comparing:

- The original plot (with visible stripes)
- A “no-floor” version (droplet shape remains, stripes disappear)
- Colored plots that reveal hidden structure inside the data

You don’t need a math background to enjoy this — the visuals speak for themselves.

---

## What is **b2**?

To create the plot, we compute a helper value called **b2** for each semiprime.  
Here’s the simple explanation:

> **b2** is calculated by taking the square root of `2 × c`, rounding it down to a whole number, and then doubling it.

This value acts as a rough estimate related to the size of the two prime factors that make up the semiprime. Using it in the formula helps reveal structure when we graph the data.

---

## Try It Yourself (No Installation Needed)

Click below to open an interactive notebook in Google Colab:

▶️ **Run in Google Colab**  
https://colab.research.google.com/drive/1DrMg5BvbQgf-hIY_zx7ivbh8LcYlaSgE

---

## How to Run Locally (Optional)

### Requirements
- Python 3.9 or later
- `numpy` and `matplotlib`

Install the required packages:

```bash
pip install numpy matplotlib
