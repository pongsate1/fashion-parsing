#include "color_map.h"

// R, G, B color map for fashionista-v1.0.
unsigned char color_mapV1[25][3] = {
  {255, 255, 255},/*background*/  
  {226, 196, 196},/*skin*/  
  {64, 32, 32},/*hair*/  
  {255, 0, 0},/*bag*/  
  {255, 191, 0},/*belt*/  
  {128, 255, 0},/*boots*/  
  {0, 255, 64},/*coat*/  
  {0, 255, 255},/*dress*/  
  {0, 64, 255},/*glasses*/  
  {128, 0, 255},/*gloves*/  
  {255, 0, 191},/*hat/headband*/  
  {255, 85, 85},/*jacket/blazer*/  
  {255, 231, 85},/*necklace*/  
  {134, 255, 85},/*pants/jeans*/  
  {85, 255, 182},/*scarf/tie*/  
  {85, 182, 255},/*shirt/blouse*/  
  {134, 85, 255},/*shoes*/  
  {255, 85, 231},/*shorts*/  
  {255, 170, 170},/*skirt*/  
  {255, 243, 170},/*socks*/  
  {194, 255, 170},/*sweater/cardigan*/  
  {170, 255, 219},/*tights/leggings*/  
  {170, 219, 255},/*top/t-shirt*/  
  {194, 170, 255},/*vest*/  
  {255, 170, 243}/*watch/bracelet*/  
};

// R, G, B color map for fashionista-v0.2.
unsigned char color_mapV2[56][3] = {
  {255, 255, 255},/*null*/  
  {226, 196, 196},/*tights*/  
  {64, 32, 32},/*shorts*/  
  {255, 0, 0},/*blazer*/  
  {255, 191, 0},/*t-shirt*/  
  {128, 255, 0},/*bag*/  
  {0, 255, 64},/*shoes*/  
  {0, 255, 255},/*coat*/  
  {0, 64, 255},/*skirt*/  
  {128, 0, 255},/*purse*/  
  {255, 0, 191},/*boots*/  
  {255, 85, 85},/*blouse*/  
  {255, 231, 85},/*jacket*/  
  {134, 255, 85},/*bra*/  
  {85, 255, 182},/*dress*/  
  {85, 182, 255},/*pants*/  
  {134, 85, 255},/*sweater*/  
  {255, 85, 231},/*shirt*/  
  {255, 170, 170},/*jeans*/  
  {255, 243, 170},/*leggings*/  
  {194, 255, 170},/*scarf*/  
  {170, 255, 219},/*hat*/  
  {170, 219, 255},/*top*/  
  {194, 170, 255},/*cardigan*/  
  {255, 170, 243},/*accessories*/  
  {39, 255, 109},/*vest*/  
  {21, 62, 188 },/*sunglasses*/  
  {193,230,134 },/*belt*/  
  {76,66,12 },/*socks*/  
  {116,187,202},/*glasses*/  
  {89,157,1 },/*intimate*/  
  {30,52,246 },/*stockings*/  
  {127,231,33},/*necklace*/  
  {158,3,199 },/*cape*/  
  {123,137,231},/*jumper*/  
  {55,194,196 },/*sweatshirt*/  
  {143,25,216 },/*suit*/  
  {166,216,102 },/*bracelet*/  
  {122,0,110 },/*heels*/  
  {45,220,52 },/*wedges*/  
  {125,214,48 },/*ring*/  
  {139,56,166 },/*flats*/  
  {120,79,218 },/*tie*/  
  {1,131,69 },/*romper*/  
  {78,37,136 },/*sandals*/  
  {50,222,164 },/*earrings*/  
  {201,50,245},/*gloves*/  
  {114,137,135 },/*sneakers*/  
  {250,149,214 },/*clogs*/  
  {31,189,92 },/*watch*/ 
  {2,79,92 },/*pumps*/  
  {163,73,214 },/*wallet*/  
  {2,171,203 },/*bodysuit*/  
  {63,195,124 },/*loafers*/  
  {44, 34, 43},/*hair*/ 
  {255, 233, 196} /*skin*/ 
};



void Visualize(const int* mask, int width, int height, unsigned char* bgr) {
  for (int i = 0, j = 0; i < width * height; ++i, j+=3) {
    int index = mask[i];
    if (index >= 56) {
      bgr[j+0] = 0;
      bgr[j+1] = 0;
      bgr[j+2] = 0;
    } else {
      bgr[j+0] = color_mapV2[index][2];
      bgr[j+1] = color_mapV2[index][1];
      bgr[j+2] = color_mapV2[index][0];
    }
  }
}