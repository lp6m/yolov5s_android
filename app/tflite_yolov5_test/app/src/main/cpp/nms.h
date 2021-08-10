//https://github.com/martinkersner/non-maximum-suppression-cpp/blob/master/nms.cpp
#include <vector>
#include <numeric>
#include <iostream>

using namespace std;

struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int class_idx;
    bbox(float x1, float y1, float x2, float y2, float conf, int class_idx) :
            x1(x1), y1(y1), x2(x2), y2(y2), conf(conf), class_idx(class_idx){
    }
};

#define Point_XMIN 0
#define Point_XMAX 1
#define Point_YMIN 2
#define Point_YMAX 3

vector<float> GetPointFromRect(const vector<bbox> &rect,
                               const int pos)
{
    vector<float> points;

    for (const auto & p: rect) {
        float point;
        if (pos == Point_XMIN) point = p.x1;
        else if (pos == Point_XMAX) point = p.x2;
        else if (pos == Point_YMIN) point = p.y1;
        else if (pos == Point_YMAX) point = p.y2;

        points.push_back(point);
    }

    return points;
}

vector<float> ComputeArea(const vector<float> & x1,
                          const vector<float> & y1,
                          const vector<float> & x2,
                          const vector<float> & y2)
{
    vector<float> area;
    auto len = x1.size();

    for (decltype(len) idx = 0; idx < len; ++idx) {
        auto tmpArea = (x2[idx] - x1[idx] + 1) * (y2[idx] - y1[idx] + 1);
        area.push_back(tmpArea);
    }

    return area;
}

vector<int> argsort_byscore(const vector<bbox> & v)
{
    // initialize original index locations
    vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](int i1, int i2) {return v[i1].conf < v[i2].conf;});

    return idx;
}

vector<float> Maximum(const float & num,
                      const vector<float> & vec)
{
    auto maxVec = vec;
    auto len = vec.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] < num)
            maxVec[idx] = num;

    return maxVec;
}

vector<float> Minimum(const float & num,
                      const vector<float> & vec)
{
    auto minVec = vec;
    auto len = vec.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] > num)
            minVec[idx] = num;

    return minVec;
}

vector<float> CopyByIndexes(const vector<float> & vec,
                            const vector<int> & idxs)
{
    vector<float> resultVec;

    for (const auto & idx : idxs)
        resultVec.push_back(vec[idx]);

    return resultVec;
}

vector<int> RemoveLast(const vector<int> & vec)
{
    auto resultVec = vec;
    resultVec.erase(resultVec.end()-1);
    return resultVec;
}

vector<float> Subtract(const vector<float> & vec1,
                       const vector<float> & vec2)
{
    vector<float> result;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        result.push_back(vec1[idx] - vec2[idx] + 1);

    return result;
}

vector<float> Multiply(const vector<float> & vec1,
                       const vector<float> & vec2)
{
    vector<float> resultVec;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        resultVec.push_back(vec1[idx] * vec2[idx]);

    return resultVec;
}

vector<float> Divide(const vector<float> & vec1,
                     const vector<float> & vec2)
{
    vector<float> resultVec;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        resultVec.push_back(vec1[idx] / vec2[idx]);

    return resultVec;
}

vector<int> WhereLarger(const vector<float> & vec,
                        const float & threshold)
{
    vector<int> resultVec;
    auto len = vec.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        if (vec[idx] > threshold)
            resultVec.push_back(idx);

    return resultVec;
}

vector<int> RemoveByIndexes(const vector<int> & vec,
                            const vector<int> & idxs)
{
    auto resultVec = vec;
    auto offset = 0;

    for (const auto & idx : idxs) {
        resultVec.erase(resultVec.begin() + idx + offset);
        offset -= 1;
    }

    return resultVec;
}


template <typename T>
vector<T> FilterVector(const vector<T> & vec,
                       const vector<int> & idxs)
{
    vector<T> resultVec;

    for (const auto & idx: idxs)
        resultVec.push_back(vec[idx]);

    return resultVec;
}


vector<bbox> nms(const vector<bbox> & candidates,
                 const float &iou_threshold)
{
    if (candidates.empty()) return vector<bbox>();

    // grab the coordinates of the bounding boxes
    auto x1 = GetPointFromRect(candidates, Point_XMIN);
    auto y1 = GetPointFromRect(candidates, Point_YMIN);
    auto x2 = GetPointFromRect(candidates, Point_XMAX);
    auto y2 = GetPointFromRect(candidates, Point_YMAX);

    // compute the area of the bounding boxes and sort the bounding
    // boxes by the bottom-right y-coordinate of the bounding box
    auto area = ComputeArea(x1, y1, x2, y2);
    auto idxs = argsort_byscore(candidates);

    int last;
    int i;
    vector<int> pick;

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last index in the indexes list and add the
        // index value to the list of picked indexes
        last = idxs.size() - 1;
        i    = idxs[last];
        pick.push_back(i);

        // find the largest (x, y) coordinates for the start of
        // the bounding box and the smallest (x, y) coordinates
        // for the end of the bounding box
        auto idxsWoLast = RemoveLast(idxs);

        auto xx1 = Maximum(x1[i], CopyByIndexes(x1, idxsWoLast));
        auto yy1 = Maximum(y1[i], CopyByIndexes(y1, idxsWoLast));
        auto xx2 = Minimum(x2[i], CopyByIndexes(x2, idxsWoLast));
        auto yy2 = Minimum(y2[i], CopyByIndexes(y2, idxsWoLast));

        // compute the width and height of the bounding box
        auto w = Maximum(0, Subtract(xx2, xx1));
        auto h = Maximum(0, Subtract(yy2, yy1));

        // compute the ratio of overlap
        auto overlap = Divide(Multiply(w, h), CopyByIndexes(area, idxsWoLast));

        // delete all indexes from the index list that have
        auto deleteIdxs = WhereLarger(overlap, iou_threshold);
        deleteIdxs.push_back(last);
        idxs = RemoveByIndexes(idxs, deleteIdxs);
    }

    return FilterVector(candidates, pick);
}

