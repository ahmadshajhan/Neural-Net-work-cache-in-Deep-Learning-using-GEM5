// matmul_food.cpp — Dense matmul on pizza/steak/sushi data for GEM5
// Compile: riscv64-linux-gnu-g++ -O0 -static -o matmul_food matmul_food.cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Matrix dimensions ---
// X: (BATCH x FEATURES) food feature matrix
// W: (FEATURES x CLASSES) weight matrix
// C = X * W: (BATCH x CLASSES) output (logits)

#define BATCH    24      // Number of images in the GEM5 batch
#define FEATURES 3072    // 32*32*3 pixels per image
#define CLASSES  3       // pizza, steak, sushi

static float X[BATCH][FEATURES];    // Input matrix
static float W[FEATURES][CLASSES];  // Weight matrix
static float C[BATCH][CLASSES];     // Output logits
static int Y[BATCH];                // Ground-truth labels for the GEM5 batch

// Read raw float32 binary file into array
int read_bin(const char *path, float *buf, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return -1; }
    int n = fread(buf, sizeof(float), count, f);
    fclose(f);
    if (n != count) {
        printf("WARNING: read %d floats, expected %d\n", n, count);
    }
    return n;
}

int read_int_bin(const char *path, int *buf, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return -1; }
    int n = fread(buf, sizeof(int), count, f);
    fclose(f);
    if (n != count) {
        printf("WARNING: read %d ints, expected %d\n", n, count);
    }
    return n;
}

// Argmax — returns class with highest logit
int argmax(float *row, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (row[i] > row[best]) best = i;
    return best;
}

int main() {
    const char *class_names[CLASSES] = {"pizza", "steak", "sushi"};

    printf("=== Neural-Network Matmul on Pizza/Steak/Sushi ===\n");
    printf("Loading feature matrix X (%d x %d)...\n", BATCH, FEATURES);

    if (read_bin("matrices/X_sub.bin", (float*)X,
                 BATCH * FEATURES) < 0) return 1;

    printf("Loading weight matrix W (%d x %d)...\n", FEATURES, CLASSES);
    if (read_bin("matrices/W.bin", (float*)W,
                 FEATURES * CLASSES) < 0) return 1;
    if (read_int_bin("matrices/y_sub.bin", (int*)Y, BATCH) < 0) return 1;

    // Zero output
    memset(C, 0, sizeof(C));

    printf("Running matrix multiply C = X @ W ...\n");

    // =============================================
    // MAC KERNEL — this is what GEM5 will profile
    // Same as a fully-connected neural network layer
    // =============================================
    for (int i = 0; i < BATCH; i++) {          // row of X (image)
        for (int j = 0; j < CLASSES; j++) {    // column of W (class)
            float acc = 0.0f;
            for (int k = 0; k < FEATURES; k++) { // inner dimension
                acc += X[i][k] * W[k][j];         // MAC operation
            }
            C[i][j] = acc;
        }
    }

    // Print first 5 predictions
    printf("\nFirst 5 predictions:\n");
    for (int i = 0; i < 5; i++) {
        int pred = argmax(C[i], CLASSES);
        printf("  Image %2d -> logits [%.4f, %.4f, %.4f] -> pred=%s actual=%s\n",
               i, C[i][0], C[i][1], C[i][2], class_names[pred], class_names[Y[i]]);
    }

    int correct = 0;
    for (int i = 0; i < BATCH; i++) {
        if (argmax(C[i], CLASSES) == Y[i]) correct++;
    }
    printf("\nSubset accuracy on GEM5 batch: %.2f%% (%d/%d)\n",
           100.0f * correct / BATCH, correct, BATCH);

    printf("\nC[0][0] = %f (sanity check)\n", C[0][0]);
    printf("Matrix multiply done!\n");
    return 0;
}
