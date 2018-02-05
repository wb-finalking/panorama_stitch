#图像拼接

### 开发环境

**VS2013**+**opencv2.4.13**+**QT5.8**

**release.rar**是可运行的发布版本

### 功能实现

一组无序的多个场景的图像进行筛选得到几组不同的场景，然后对不同的场景进行图像拼接

* 对输入的一组图像进行ORB特征提取和匹配和匹配。
* 利用图像间的特征匹配对在相同场景下有重叠区域的图像进行分组，并剔除不与任何其他图像重叠的图像 。
* 对每组图像进行全景图拼接，并使用拉普拉斯金字塔的方法消除接缝处的视觉差异。
### 输入
通过窗口菜单 **<u>File path</u>** 选项选择图片存放路径
### 输出
每组场景拼接结果显示在窗口，并通过next和pre按钮实现不同场景切换，最后可以通过菜单 **<u>Save</u>** 选项选择保存路径将每组场景结果保存。

### 结果展示

![](https://github.com/wb-finalking/panorama_stitch/blob/master/CV_hw_qt/res/res0.jpg?raw=true)

![](https://github.com/wb-finalking/panorama_stitch/blob/master/CV_hw_qt/res/res2.jpg?raw=true)





### 详细设计

####1>特征提取

通过使用opencv自带的OrbFeatureDetector类方法得到每幅图像的特征点信息，使用OrbDescriptorExtractor类方法得到每个特征点的特征描述。

```C++
//提取特征点
std::vector<cv::KeyPoint> keypoints;
cv::OrbFeatureDetector  orbDetector(3000);
orbDetector.detect(image1, keypoints);`

//计算每个特征点的特征描述
cv::OrbDescriptorExtractor  orbDescriptor;
cv::Mat imageDesc;
SurfDescriptor.compute(image1, feature, imageDesc);
```
#### 2>特征点匹配

使用opencv自带的Fast Library for Approximate Nearest Neighbors库为一副图像的特征点描述矩阵imageDesc构建以汉明距离的Locality-Sensitive Hashing索引，为另一副图像的每一个特征点描述寻找此索引下最近的2个特征，通过判断2个距离的关系决定是否接受此特征点对。

```C++
//构建Locality-Sensitive Hashing索引
cv::flann::Index flannIndex(imageDesc1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

//为另一副图像的每一个特征点描述寻找此索引下最近的2个特征点
cv::Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, cv::flann::SearchParams());
	
//判断最近的特征距离是否是第二个最近距离的0.4倍，满足条件就接受此特征点与最近特征点组成的点对
vector<cv::DMatch> GoodMatchePoints;
for (int i = 0; i < matchDistance.rows; i++)
{
  if (matchDistance.at<float>(i, 0) < 0.4 * matchDistance.at<float>(i, 1))
  {
    cv::DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
    GoodMatchePoints.push_back(dmatches);
  }
}
```
#### 3>场景分组

得到不同图像之间的特征点配对数目，然后根据设定的阈值判断两幅图像是否有重叠即是否属于同一个场景，得到图像之间的邻接矩阵，通过邻接矩阵寻找连通域得到图像的不同场景分组。

```C++
//计算邻接矩阵need_stitching并存储不同的特征匹配对在GoodMatches
for (int i = 0; i < src_imgs.size(); i++)
{
  for (int j = i+1; j < src_imgs.size(); j++)
  {
    if (i == j)
      continue;
    vector<cv::DMatch> GoodMatchePoints;
    GoodMatchePoints = orbpairs(descriptor[i], descriptor[j]);

    if (GoodMatchePoints.size() >= 20) 
    {
      need_stitching[i][j] = true;
      need_stitching[j][i] = true;
      cout << "Image " << i << " and " << j << " are adjacent." << endl;
      matching_index[i].push_back(j);
      matching_index[j].push_back(i);
      Pair pair(i,j);
      GoodMatches.insert(make_pair(pair, GoodMatchePoints));
    }
  }
}
```

```C++
//通过need_stitching寻找连通域并存储在connected_domain
vector<vector<int> > connected_domain;
vector<bool> unAssigned(src_imgs_size, TRUE);
while (1)
{
  vector<int> domain;
  //寻找初始点
  int start_index = -1;
  for (int i = 0; i < unAssigned.size(); i++)
  {
    if (unAssigned[i] == TRUE)
    {
      start_index = i;
      break;
    }
  }
  //如果没有初始点则退出
  if (start_index == -1)
    break;

  //通过广度优先搜索寻找一个连通域内的所有点
  queue<int> q;
  q.push(start_index);
  while (!q.empty())
  {
    int cur_index = q.front();
    q.pop();
    unAssigned[cur_index] = FALSE;
    domain.push_back(cur_index);
    for (int i = 0; i < matching_index[cur_index].size(); i++)
    {
      if (unAssigned[matching_index[cur_index][i]])
        q.push(matching_index[cur_index][i]);
    }
  }
  connected_domain.push_back(domain);
}
```

#### 4>计算扭曲变换关系矩阵H

通过特征点匹配对计算两幅图像之间的扭曲变换关系矩阵 2*4 H。

x = h1 * x + h2 * y + h3 * x * y + h4

y = h5 * x + h6 * y + h7 * x * y + h8

通过RANSAC（随机抽样一致性）算法剔除一些匹配对

```C++
//迭代剔除局外匹配对
while (iterations--) 
{
  //随机得到4对匹配对存储在random_pairs
  vector<point_pair> random_pairs;
  set<int> seleted_indexs;
  for (int i = 0; i < NUM_OF_PAIR; i++) 
  {
    int index = random(0, pairs.size() - 1);
    while (seleted_indexs.find(index) != seleted_indexs.end()) 
    {
      index = random(0, pairs.size() - 1);
    }
    seleted_indexs.insert(index);

    random_pairs.push_back(pairs[index]);
  }

  //通过随机得到的匹配对计算变换矩阵 H
  Parameters H = getHomographyFromPoingPairs(random_pairs);

  //通过 H 计算匹配对内变换之后的距离差，然后判断距离是否在阈值范围内进行剔除得到cur_inliner_indexs
  vector<int> cur_inliner_indexs = getIndexsOfInliner(pairs, H, seleted_indexs);
  //max_inliner_indexs与cur_inliner_indexs做个数大小比较取个数较大者做记录
  if (cur_inliner_indexs.size() > max_inliner_indexs.size()) 
  {
    max_inliner_indexs = cur_inliner_indexs;
  }
}
```
通过CImg库求解 H
```C++
CImg<double> A(4, calc_size, 1, 1, 0);
CImg<double> b(1, calc_size, 1, 1, 0);

for (int i = 0; i < calc_size; i++) {
  int cur_index = inliner_indexs[i];

  A(0, i) = pairs[cur_index].a.pt.y;
  A(1, i) = pairs[cur_index].a.pt.x;
  A(2, i) = pairs[cur_index].a.pt.x * pairs[cur_index].a.pt.y;
  A(3, i) = 1;

  b(0, i) = pairs[cur_index].b.pt.y;
}

CImg<double> x1 = b.get_solve(A);

for (int i = 0; i < calc_size; i++) 
{
  int cur_index = inliner_indexs[i];
  b(0, i) = pairs[cur_index].b.pt.x;
}

CImg<double> x2 = b.get_solve(A);

H = Parameters(x1(0, 0), x1(0, 1), x1(0, 2), x1(0, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3));
```

#### 5>扭曲变换

通过变换矩阵 H 将图像进行扭曲变换，在计算图像新坐标对应的像素值时采取双线性插值的方法。

```C++
// x ，y 皆是float类型，要通过插值找到对应的像素值
int x_pos = floor(x);
float x_u = x - x_pos;
int xb = (x_pos < image.rows - 1) ? x_pos + 1 : x_pos;

int y_pos = floor(y);
float y_v = y - y_pos;
int yb = (y_pos < image.cols - 1) ? y_pos + 1 : y_pos;

cv::Vec3b P1 = image.at<cv::Vec3b>(x_pos, y_pos) * (1 - x_u) + image.at<cv::Vec3b>(xb, y_pos) * x_u;
cv::Vec3b P2 = image.at<cv::Vec3b>(x_pos, yb) * (1 - x_u) + image.at<cv::Vec3b>(xb, yb) * x_u;

cv::Vec3b new = P1 * (1 - y_v) + P2 * y_v;
```

通过变换矩阵 H 计算新的图像每个坐标对应的像素值

```C++
//cv::Mat dst 变换后的图像
//cv::Mat src 变换前的图像
for (int dst_x = 0; dst_x < dst.rows; dst_x++) 
{
  for (int dst_y = 0; dst_y < dst.cols; dst_y++) 
  {
    int src_x = getXAfterWarping(dst_x , dst_y , H);
    int src_y = getYAfterWarping(dst_x , dst_y , H);

    if (src_x >= 0 && src_x < src.rows && src_y >= 0 && src_y < src.cols) 
      dst.at<cv::Vec3b>(dst_x, dst_y) = bilinear_interpolation(src, src_x, src_y);
  }
}
```

#### 6>图像融合

通过重叠部分判断得到mask矩阵来拼接两张图片，期间通过拉普拉斯金字塔对拼接出平滑处理。

得到mask矩阵

```C++
//计算图片a的重心位置和，a 和 b 重叠部分的重心
//sum_a_x或sum_a_y 是累计坐标值，然后除以个数 a_n 可以得到重心
//sum_overlap_x或sum_overlap_y 是累计坐标值，然后除以个数 overlap_n 可以得到重心
if (a.rows > a.cols) {
  for (int x = 0; x < a.rows; x++) {
    if (!isEmpty(a, x, a.cols / 2)) {
      sum_a_x += x;
      a_n++;
    }
    if (!isEmpty(a, x, a.cols / 2) && !isEmpty(b, x, a.cols / 2)) {
      sum_overlap_x += x;
      overlap_n++;
    }
  }
}
else {
  for (int y = 0; y < a.cols; y++) {
    if (!isEmpty(a, a.rows / 2, y)) {
      sum_a_y += y;
      a_n++;
    }
    if (!isEmpty(a, a.rows / 2, y) && !isEmpty(b, b.rows / 2, y)) {
      sum_overlap_y += y;
      overlap_n++;
    }
  }
}
	
mask = cv::Mat(a.rows, a.cols, CV_32FC3, cv::Scalar(0, 0, 0));
//通过比较重心位置得到mask矩阵
if (a.rows > a.cols) {
  if (sum_a_x / a_n < sum_overlap_x / overlap_n) {
    for (int x = 0; x < sum_overlap_x / overlap_n; x++) {
      for (int y = 0; y < a.cols; y++) {
        mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
      }
    }
  }
  else {
    for (int x = sum_overlap_x / overlap_n + 1; x < a.rows; x++) {
      for (int y = 0; y < a.cols; y++) {
        mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
      }
    }
  }
}
else {
  if (sum_a_y / a_n < sum_overlap_y / overlap_n) {
    for (int x = 0; x < a.rows; x++) {
      for (int y = 0; y < sum_overlap_y / overlap_n; y++) {
        mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
      }
    }
  }
  else {
    for (int x = 0; x < a.rows; x++) {
      for (int y = sum_overlap_y / overlap_n; y < a.cols; y++) {
        mask[0].at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
      }
    }
  }
}
```

构建拉普拉斯金字塔然后将每一层的拼接结果累加则得到最后结果

```C++
//首先构建降采样金字塔
for (int i = 1; i < n_level; i++) {
  cv::resize(a_pyramid[i-1], a_pyramid[i], cv::Size(a_pyramid[i-1].cols / 2, a_pyramid[i-1].rows / 2));
  cv::resize(b_pyramid[i-1], b_pyramid[i], cv::Size(b_pyramid[i-1].cols / 2, b_pyramid[i-1].rows / 2));

  cv::resize(mask[i-1], mask[i], cv::Size(mask[i-1].cols / 2, mask[i-1].rows / 2));
}

//构建拉普拉斯金字塔
for (int i = 0; i < n_level - 1; i++)
{
  cv::Mat tmp;
  cv::resize(a_pyramid[i+1], tmp, cv::Size(a_pyramid[i].cols, a_pyramid[i].rows));
  a_pyramid[i] = a_pyramid[i] - tmp;
  cv::resize(b_pyramid[i + 1], tmp, cv::Size(b_pyramid[i].cols, b_pyramid[i].rows));
  b_pyramid[i] = b_pyramid[i] - tmp;
}

//计算每一层的拼接
vector<cv::Mat > blend_pyramid(n_level);
for (int i = 0; i < n_level; i++) 
{
  blend_pyramid[i] = cv::Mat(a_pyramid[i].rows, a_pyramid[i].cols, CV_32FC3,cv::Scalar(0,0,0));

  for (int x = 0; x < blend_pyramid[i].rows; x++) {
    for (int y = 0; y < blend_pyramid[i].cols; y++) {
      blend_pyramid[i].at<cv::Vec3f>(x, y)[0] = a_pyramid[i].at<cv::Vec3f>(x, y)[0] * 			mask[i].at<cv::Vec3f>(x, y)[0] + b_pyramid[i].at<cv::Vec3f>(x, y)[0] * (1.0 - 				mask[i].at<cv::Vec3f>(x, y)[0]);
      blend_pyramid[i].at<cv::Vec3f>(x, y)[1] = a_pyramid[i].at<cv::Vec3f>(x, y)[1] * 			mask[i].at<cv::Vec3f>(x, y)[1] + b_pyramid[i].at<cv::Vec3f>(x, y)[1] * (1.0 - 				mask[i].at<cv::Vec3f>(x, y)[1]);
      blend_pyramid[i].at<cv::Vec3f>(x, y)[2] = a_pyramid[i].at<cv::Vec3f>(x, y)[2] * 			mask[i].at<cv::Vec3f>(x, y)[2] + b_pyramid[i].at<cv::Vec3f>(x, y)[2] * (1.0 - 				mask[i].at<cv::Vec3f>(x, y)[2]);
    }
  }
}

//将每一层累计求和得到拼接结果
cv::Mat res = blend_pyramid[n_level-1];
for (int i = n_level - 2; i >= 0; i--) 
{
  cv::resize(res, res, cv::Size(blend_pyramid[i].cols, blend_pyramid[i].rows));
  res = res + blend_pyramid[i];
}
```

### 附录

![](https://github.com/wb-finalking/panorama_stitch/blob/master/res0.jpg?raw=true)

![](https://github.com/wb-finalking/panorama_stitch/blob/master/res2.jpg?raw=true)
