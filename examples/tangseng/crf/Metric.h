#pragma once

float AllPixelAcc(const int* mask, const int* gt, int width, int height);
float FgdPixelAcc(const int* mask, const int* gt, int width, int height, int bgd_id = 0);