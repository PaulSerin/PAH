#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Paramètres du filtre
#define SIGMA_S 3.0      // Influence de la distance spatiale
#define SIGMA_R 50.0     // Influence de la différence d'intensité
#define WINDOW_SIZE 5    // Taille de la fenêtre de voisinage (5x5)

// Début du programme principal
int main() {
    // Ouvrir l'image BMP en lecture binaire
    FILE *f_in = fopen("../Images/lena.bmp", "rb");
    if (!f_in) { perror("Erreur ouverture"); exit(1); }

    // Lecture des 54 octets d'en-tête BMP
    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, f_in);

    // Récupération des infos essentielles dans l'entête BMP
    int width = *(int*)&header[18];
    int height = *(int*)&header[22];
    int bitCount = *(short*)&header[28];
    printf("Image : %d x %d - %d bits\n", width, height, bitCount);

    // Vérification que l'image est bien en 8 bits niveau de gris
    if (bitCount != 8) {
        printf("Image NON 8 bits, non supportée.\n");
        exit(1);
    }

    // Lecture de la palette (256 couleurs * 4 octets = 1024 octets)
    unsigned char palette[1024];
    fread(palette, sizeof(unsigned char), 1024, f_in);

    // Lecture des pixels avec gestion du padding
    int row_padded = (width + 3) & (~3);  // Alignement sur 4 octets
    unsigned char *input = (unsigned char *)malloc(width * height);
    unsigned char *row = (unsigned char *)malloc(row_padded);

    // Lecture ligne par ligne
    for (int i = 0; i < height; i++) {
        fread(row, sizeof(unsigned char), row_padded, f_in);
        for (int j = 0; j < width; j++) {
            input[i * width + j] = row[j];  // On copie uniquement les pixels sans le padding
        }
    }
    fclose(f_in);

    // Préparation du tableau de sortie
    unsigned char *output = (unsigned char *)malloc(width * height);

    // Pré-calcul du noyau spatial (gaussien sur la distance)
    int radius = WINDOW_SIZE / 2;
    double spatial[WINDOW_SIZE][WINDOW_SIZE];
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            spatial[i + radius][j + radius] = exp(-(i * i + j * j) / (2 * SIGMA_S * SIGMA_S));
        }
    }

    // DÉMARRAGE DU TIMER
    clock_t start = clock();

    // Application du filtre bilatéral
    printf("Filtrage en cours...\n");
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double sum_weight = 0.0;
            double sum_intensity = 0.0;
            unsigned char center = input[y * width + x];  // Pixel central

            // Parcours de la fenêtre autour du pixel central
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    unsigned char neighbor = input[(y + i) * width + (x + j)];
                    double diff = neighbor - center; // Différence d'intensité

                    // Calcul des poids spatial et intensité
                    double weight = spatial[i + radius][j + radius] * exp(-(diff * diff) / (2 * SIGMA_R * SIGMA_R));

                    sum_weight += weight;
                    sum_intensity += weight * neighbor;
                }
            }
            // On calcule la nouvelle valeur du pixel
            output[y * width + x] = (unsigned char)(sum_intensity / sum_weight);
        }
    }

    // ARRÊT DU TIMER
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps CPU pour le filtrage : %.6f secondes\n", cpu_time);

    // Écriture de l'image filtrée dans un nouveau BMP
    FILE *f_out = fopen("../Images/lena_filtered_simple.bmp", "wb");

    // On réécrit l'en-tête et la palette
    fwrite(header, sizeof(unsigned char), 54, f_out);
    fwrite(palette, sizeof(unsigned char), 1024, f_out);

    // Écriture des pixels avec gestion du padding
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fputc(output[i * width + j], f_out);
        }
        // Remplissage du padding à 0
        for (int p = width; p < row_padded; p++) fputc(0, f_out);
    }

    fclose(f_out);

    // Libération de la mémoire
    free(input);
    free(output);
    free(row);

    printf("Filtrage terminé ! Résultat : lena_filtered_simple.bmp\n");
    return 0;
}