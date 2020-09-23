#include <stdio.h>
#include <tigergrav/expansion.hpp>

static constexpr size_t map2[3][3] = { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } };
static constexpr size_t map3[3][3][3] = { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 }, { 5,
		8, 9 } } };
static constexpr size_t map4[3][3][3][3] = { { { { 0, 1, 2 }, { 1, 3, 4 }, { 2, 4, 5 } }, { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 2, 4, 5 }, { 4, 7, 8 },
		{ 5, 8, 9 } } }, { { { 1, 3, 4 }, { 3, 6, 7 }, { 4, 7, 8 } }, { { 3, 6, 7 }, { 6, 10, 11 }, { 7, 11, 12 } },
		{ { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } } }, { { { 2, 4, 5 }, { 4, 7, 8 }, { 5, 8, 9 } }, { { 4, 7, 8 }, { 7, 11, 12 }, { 8, 12, 13 } }, {
		{ 5, 8, 9 }, { 8, 12, 13 }, { 9, 13, 14 } } } };

static const double efs[LP + 1] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 5.00000000e-01, 1.00000000e+00, 1.00000000e+00,
		5.00000000e-01, 1.00000000e+00, 5.00000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 1.00000000e+00, 5.00000000e-01,
		1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 1.66666667e-01, 2.50000000e-01, 5.00000000e-01,
		2.50000000e-01, 1.66666667e-01, 5.00000000e-01, 5.00000000e-01, 1.66666667e-01, 4.16666667e-02, 1.66666667e-01, 2.50000000e-01, 1.66666667e-01,
		4.16666667e-02, 0.0 };
const expansion<double> &expansion_factor = *reinterpret_cast<const expansion<double>*>(efs);

int main() {

////	Dfour() = fma(hpart(), co, Dfour());
//	printf("Dfour[0] = fma(hpart[0], co, Dfour[0]);\n");
//	for (int a = 0; a < NDIM; a++) {
////		Dfour(a) = fma(hpart(a), so, Dfour(a));
//		printf("Dfour[%i] = fma(hpart[%i], so, Dfour[%i]);\n", 1 + a, 1 + a, 1 + a);
//		for (int b = 0; b <= a; b++) {
//			//		Dfour(a, b) = fma(hpart(a, b), co, Dfour(a, b));
//			printf("Dfour[%i] = fma(hpart[%i], co, Dfour[%i]);\n", 4 + map2[a][b], 4 + map2[a][b], 4 + map2[a][b]);
//			for (int c = 0; c <= b; c++) {
////				Dfour(a, b, c) = fma(hpart(a, b, c), so, Dfour(a, b, c));
//				printf("Dfour[%i] = fma(hpart[%i], so, Dfour[%i]);\n", 10 + map3[a][b][c], 10 + map3[a][b][c], 10 + map3[a][b][c]);
//				for (int d = 0; d <= c; d++) {
//					printf("Dfour[%i] = fma(hpart[%i], co, Dfour[%i]);\n", 20 + map4[a][b][c][d], 20 + map4[a][b][c][d], 20 + map4[a][b][c][d]);
//				}
//			}
//		}
//	}
//
	printf("const auto dx0dx0 = dx[0] * dx[0];\n"
			"const auto dx0dx1 = dx[0] * dx[1];\n"
			"const auto dx0dx2 = dx[0] * dx[2];\n"
			"const auto dx1dx1 = dx[1] * dx[1];\n"
			"const auto dx1dx2 = dx[1] * dx[2];\n"
			"const auto dx2dx2 = dx[2] * dx[2];\n");
	int flop = 0;
	printf("D[0] = d0;\n");
	for (int a = 0; a < NDIM; a++) {
		flop += 1;
		printf("D[%i] = fma( dx[%i], d1, D[%i]);\n", 1 + a, a, 1 + a);
		for (int b = 0; b <= a; b++) {
			flop += 1;
			printf("D[%i] = fma( dx%idx%i, d2, D[%i]);\n", 4 + (int) map2[a][b], a, b, 4 + (int) map2[a][b]);
			for (int c = 0; c <= b; c++) {
				flop += 1;
				printf("dxadxbdxc = dx%idx%i * dx[%i];\n", a, b, c);
				flop += 1;
				printf("D[%i] = fma( dxadxbdxc, d3, D[%i]);\n", 10 + (int) map3[a][b][c], 10 + (int) map3[a][b][c]);
				for (int d = 0; d <= c; d++) {
					flop += 2;
					printf("D[%i] = fma( dxadxbdxc * dx[%i], d4, D[%i]);\n", 20 + (int) map4[a][b][c][d], d, 20 + (int) map4[a][b][c][d]);
				}
			}
		}
	}
//
//	printf("D[0] = d0;\n");
//	for (int a = 0; a < NDIM; a++) {
//		flop += 1;
//		printf("D[%i] = dx[%i] * d1;\n", 1 + a, a);
//		for (int b = 0; b <= a; b++) {
//			flop += 1;
//			printf("D[%i] = dx%idx%i * d2;\n", 4 + (int) map2[a][b], a, b);
//			for (int c = 0; c <= b; c++) {
//				flop += 1;
//				printf("dxadxbdxc = dx%idx%i * dx[%i];\n", a, b, c);
//				flop += 1;
//				printf("D[%i] = dxadxbdxc * d3;\n", 10 + (int) map3[a][b][c]);
//				for (int d = 0; d <= c; d++) {
//					flop += 2;
//					printf("D[%i] = dxadxbdxc * dx[%i] * d4;\n", 20 + (int) map4[a][b][c][d], d);
//				}
//			}
//		}
//	}
	flop += 6;
	printf("\n");
	printf("const auto dx0d2 = dx[0] * d2;\n"
			"const auto dx1d2 = dx[1] * d2;\n"
			"const auto dx2d2 = dx[2] * d2;\n");
	for (int a = 0; a < NDIM; a++) {
		flop += 1;
		printf("D[%i] += d1;\n", 4 + (int) map2[a][a]);
		flop += 2;
		printf("D[%i] = fma(float(3), dx%id2, D[%i]);\n", 10 + (int) map3[a][a][a], a, 10 + (int) map3[a][a][a]);
		flop += 3;
		printf("D[%i] = fma(float(6)*dx%idx%i, d3, D[%i]);\n", 20 + (int) map4[a][a][a][a], a, a, 20 + (int) map4[a][a][a][a]);
		flop += 2;
		printf("D[%i] = fma(float(2), d2, D[%i]);\n", 20 + (int) map4[a][a][a][a], 20 + (int) map4[a][a][a][a]);
		flop += 1;
		printf("D[%i] += d2;\n", 20 + (int) map4[a][a][a][a]);
		for (int b = 0; b < a; b++) {
			flop += 1;
			printf("threedxadxb = float(3) * dx%idx%i;\n", a, b);
			flop += 1;
			printf("D[%i] += dx%id2;\n", 10 + (int) map3[a][a][b], b);
			flop += 1;
			printf("D[%i] += dx%id2;\n", 10 + (int) map3[a][b][b], a);
			flop += 2;
			printf("D[%i] = fma(threedxadxb, d3, D[%i]);\n", 20 + (int) map4[a][a][a][b], 20 + (int) map4[a][a][a][b]);
			flop += 2;
			printf("D[%i] = fma(threedxadxb, d3, D[%i]);\n", 20 + (int) map4[a][b][b][b], 20 + (int) map4[a][b][b][b]);
			flop += 1;
			printf("D[%i] += d2;\n", 20 + (int) map4[a][a][b][b]);
			flop += 2;
			printf("D[%i] = fma(dx%idx%i, d3, D[%i]);\n", 20 + (int) map4[a][a][b][b], b, b, 20 + (int) map4[a][a][b][b]);
			flop += 2;
			printf("D[%i] = fma(dx%idx%i, d3, D[%i]);\n", 20 + (int) map4[a][a][b][b], a, a, 20 + (int) map4[a][a][b][b]);
			for (int c = 0; c < b; c++) {
				flop += 2;
				printf("D[%i] = fma(dx%idx%i, d3, D[%i]);\n", 20 + (int) map4[a][a][b][c], b, c, 20 + (int) map4[a][a][b][c]);
				flop += 2;
				printf("D[%i] = fma(dx%idx%i, d3, D[%i]);\n", 20 + (int) map4[a][b][c][c], a, b, 20 + (int) map4[a][b][c][c]);
				flop += 2;
				printf("D[%i] = fma(dx%idx%i,  d3, D[%i]);\n", 20 + (int) map4[a][b][b][c], a, c, 20 + (int) map4[a][b][b][c]);
			}
		}
	}

	printf("FLOP = %i\n", flop);

//	int flop = 0;
//	for (int a = 0; a < 3; a++) {
//		for (int b = a; b < 3; b++) {
////			L0() = fma(M2(a, b) * D(a, b), expansion_factor(a, b), L0());						// 36
//			if (expansion_factor(a, b) == 1.0) {
//				flop += 2;
//				printf("L[0] = fma( M2[%i], D[%i], L[0]);\n", 1 + (int) (int) map2[a][b], 4 + (int) (int) map2[a][b]);
//			} else {
//				flop += 3;
//				printf("L[0] = fma( M2[%i], D[%i] * float(%.9e), L[0]);\n", 1 + (int) (int) map2[a][b], 4 + (int) (int) map2[a][b], expansion_factor(a, b));
//			}
//			for (int c = b; c < 3; c++) {
////				L0() = fma(-M2(a, b, c) * D(a, b, c), expansion_factor(a, b, c), L0());			// 60
//				if (expansion_factor(a, b, c) == 1.0) {
//					flop += 3;
//					printf("L[0] = fma( -M2[%i], D[%i], L[0]);\n", 7 + (int) map3[a][b][c], 10 + (int) map3[a][b][c]);
//				} else {
//					flop += 4;
//					printf("L[0] = fma( -M2[%i], D[%i] * float(%.9e), L[0]);\n", 7 + (int) map3[a][b][c], 10 + (int) map3[a][b][c], expansion_factor(a, b, c));
//				}
//			}
//		}
//	}
//
//	for (int a = 0; a < 3; a++) {
////		auto &La = L(a);
//		for (int b = 0; b < 3; b++) {
//			for (int c = b; c < 3; c++) {
////				La = fma(M2(c, b) * D(a, b, c), expansion_factor(c, b), La);				// 108
//				if (expansion_factor(b, c) == 1.0) {
//					flop += 2;
//					printf("L[%i] = fma( M2[%i], D[%i], L[%i]);\n", 1 + a, 1 + (int) map2[c][b], 10 + (int) map3[a][b][c], 1 + a);
//				} else {
//					flop += 3;
//					printf("L[%i] = fma( M2[%i], D[%i] * float(%.9e), L[%i]);\n", 1 + a, 1 + (int) map2[c][b], 10 + (int) map3[a][b][c], expansion_factor(b, c),
//							1 + a);
//				}
//				for (int d = c; d < 3; d++) {
//					//				La = fma(-M2(b, c, d) * D(a, b, c, d), expansion_factor(b, c, d), La);	//180
//					if (expansion_factor(b, c, d) == 1.0) {
//						flop += 3;
//						printf("L[%i] = fma( -M2[%i], D[%i], L[%i]);\n", 1 + a, 7 + (int) map3[b][c][d], 20 + (int) map4[a][b][c][d], 1 + a);
//					} else {
//						flop += 4;
//						printf("L[%i] = fma( -M2[%i], D[%i] * float(%.9e), L[%i]);\n", 1 + a, 7 + (int) map3[b][c][d], 20 + (int) map4[a][b][c][d],
//								expansion_factor(b, c, d), 1 + a);
//					}
//				}
//			}
//		}
//	}
//
//	for (int a = 0; a < 3; a++) {
//		for (int b = a; b < 3; b++) {
////			auto &Lab = L(a, b);
//			for (int c = 0; c < 3; c++) {
//				for (int d = c; d < 3; d++) {
////					Lab = fma(M2(c, d) * D(a, b, c, d), expansion_factor(c, d), Lab);	 // 216
//					if (expansion_factor(c, d) == 1.0) {
//						flop += 2;
//						printf("L[%i] = fma( M2[%i], D[%i], L[%i]);\n", 4 + (int) map2[a][b], 1 + (int) map2[c][d], 20 + (int) map4[a][b][c][d],
//								4 + (int) map2[a][b]);
//					} else {
//						flop += 3;
//						printf("L[%i] = fma( M2[%i], D[%i] * float(%.9e), L[%i]);\n", 4 + (int) map2[a][b], 1 + (int) map2[c][d], 20 + (int) map4[a][b][c][d],
//								expansion_factor(c, d), 4 + (int) map2[a][b]);
//					}
//				}
//			}
//		}
//	}
//	printf("%i\n", flop);
	return 0;
}
