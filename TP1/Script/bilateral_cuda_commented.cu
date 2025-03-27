#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cuda.h>  // Nécessaire pour la gestion CUDA et les timers

// Paramètres du filtre bilatéral
#define SIGMA_S 3.0        // Écart-type spatial : contrôle l'influence de la distance
#define SIGMA_R 50.0       // Écart-type d'intensité : contrôle l'influence de la différence d'intensité
#define WINDOW_SIZE 5      // Taille de la fenêtre de voisinage


// CUDA Kernel : 1 thread = 1 pixel traité
__global__ void bilateral_filter_kernel(unsigned char *input, unsigned char *output, int width, int height, int radius, double sigma_s, double sigma_r) {
    // Calcul de la position du pixel que ce thread doit traiter
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // On évite de traiter les pixels sur les bords (pour ne pas sortir du tableau)
    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double sum_weight = 0.0;
        double sum_intensity = 0.0;

        // Pixel central de la fenêtre
        unsigned char center = input[y * width + x];

        // Parcours de la fenêtre autour du pixel (5x5 dans ce cas)
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                unsigned char neighbor = input[(y + i) * width + (x + j)];

                // Calcul du poids spatial basé sur la distance
                double spatial = exp(-(i * i + j * j) / (2 * sigma_s * sigma_s));

                // Calcul du poids de l'intensité (différence de gris entre voisin et centre)
                double diff = neighbor - center;
                double range = exp(-(diff * diff) / (2 * sigma_r * sigma_r));

                // Poids total
                double weight = spatial * range;

                // Accumulation pondérée
                sum_weight += weight;
                sum_intensity += weight * neighbor;
            }
        }
        // Mise à jour du pixel de sortie avec la moyenne pondérée
        output[y * width + x] = (unsigned char)(sum_intensity / sum_weight);
    }
}

int main() {
    // === LECTURE DE L'IMAGE BMP ===
    FILE *f_in = fopen("../Images/lena.bmp", "rb");
    if (!f_in) { perror("Erreur ouverture"); exit(1); }

    // Lecture de l'en-tête BMP (54 octets)
    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, f_in);

    // Extraction des dimensions et du nombre de bits par pixel
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int bitCount = *(short*)&header[28];
    printf("Image : %d x %d - %d bits\n", width, height, bitCount);

    // Vérification que l'image est bien en 8 bits (niveaux de gris)
    if (bitCount != 8) {
        printf("Image NON 8 bits, non supportée.\n");
        exit(1);
    }

    // Lecture de la palette (1024 octets = 256 * 4)
    unsigned char palette[1024];
    fread(palette, sizeof(unsigned char), 1024, f_in);

    // Calcul du padding (les lignes sont alignées sur 4 octets)
    int row_padded = (width + 3) & (~3);

    // Allocation pour stocker l'image
    unsigned char *input = (unsigned char *)malloc(width * height);
    unsigned char *row = (unsigned char *)malloc(row_padded);

    // Lecture des pixels, ligne par ligne
    for (int i = 0; i < height; i++) {
        fread(row, sizeof(unsigned char), row_padded, f_in);
        for (int j = 0; j < width; j++) {
            input[i * width + j] = row[j];
        }
    }
    fclose(f_in);
    free(row);  // Libère la mémoire temporaire utilisée pour les lignes

    // === ALLOCATION DE LA MEMOIRE DE SORTIE ===
    unsigned char *output = (unsigned char *)malloc(width * height);

    // === ALLOCATION SUR LE GPU ===
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);    // Image d'entrée sur le GPU
    cudaMalloc(&d_output, width * height);   // Image filtrée sur le GPU

    // Copie de l'image CPU -> GPU
    cudaMemcpy(d_input, input, width * height, cudaMemcpyHostToDevice);

    // === INITIALISATION DU TIMER CUDA ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Démarrage du timer GPU

    // === CONFIGURATION DES BLOCS ET GRILLE POUR CUDA ===
    dim3 blockDim(16, 16);  // Chaque bloc contient 16x16 threads
    dim3 gridDim = (32, 32);  // Nombre de blocs
    int radius = WINDOW_SIZE / 2;  // Rayon de la fenêtre autour du pixel

    // dim3 gridDim((width + blockDim.x - 1) / blockDim.x,    // Grille qui s'adapte à la taille de l'image
    //          (height + blockDim.y - 1) / blockDim.y);

    // === LANCEMENT DU KERNEL SUR LE GPU ===
    bilateral_filter_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, radius, SIGMA_S, SIGMA_R);
    cudaDeviceSynchronize(); // Attend la fin de tous les threads GPU

    // === ARRET DU TIMER ET CALCUL DU TEMPS ===
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Temps GPU pour le filtrage : %.3f ms\n", milliseconds);

    // === RECUPERATION DU RESULTAT DU GPU VERS LE CPU ===
    cudaMemcpy(output, d_output, width * height, cudaMemcpyDeviceToHost);

    // === SAUVEGARDE DU RESULTAT DANS UN FICHIER BMP ===
    FILE *f_out = fopen("lena_filtered_cuda.bmp", "wb");
    if (!f_out) { perror("Erreur d'ouverture du fichier de sortie"); exit(1); }

    // Écriture de l'entête BMP et de la palette
    fwrite(header, sizeof(unsigned char), 54, f_out);
    fwrite(palette, sizeof(unsigned char), 1024, f_out);

    // Écriture des pixels filtrés avec gestion du padding
    for (int i = 0; i < height; i++) {
        fwrite(&output[i * width], sizeof(unsigned char), width, f_out);
        // Remplissage du padding (si nécessaire)
        for (int p = width; p < row_padded; p++) fputc(0, f_out);
    }

    fclose(f_out); // Fermeture du fichier de sortie

    // === LIBERATION DES RESSOURCES ===
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Filtrage CUDA terminé ! Résultat : lena_filtered_cuda.bmp\n");
    return 0;
}