#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <string>

struct SamplePoint {
    double t;   // 秒
    cv::Point2d px; // 像素坐标
};

// 检测蓝色弹丸
cv::Point2d detectBlueProjectile(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar lower_blue(80, 20, 20);    
    cv::Scalar upper_blue(150, 255, 255); 
    cv::Mat mask;
    cv::inRange(hsv, lower_blue, upper_blue, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    
    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return {-1, -1};
    }
    
    // 找最大轮廓
    double max_area = 0;
    int max_idx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area && area > 3) {
            max_area = area;
            max_idx = i;
        }
    }
    
    if (max_idx == -1) {
        return {-1, -1};
    }
    
    // 计算质心
    cv::Moments moments = cv::moments(contours[max_idx]);
    if (moments.m00 == 0) {
        return {-1, -1};
    }
    
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;
    
    return {cx, cy};
}

// 弹道模型
struct TrajModelCost {
    TrajModelCost(double t_i, double x_i, double y_i, double x0, double y0)
        : t(t_i), xi(x_i), yi(y_i), x0(x0), y0(y0) {}
    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        const T vx0 = params[0];
        const T vy0 = params[1];
        const T g   = params[2];
        const T k   = params[3];

        const T expkt = ceres::exp(-k * T(t));
        const T x = T(x0) + (vx0 / k) * (T(1.0) - expkt);
        const T y = T(y0) + ((vy0 + g / k) / k) * (T(1.0) - expkt) - (g / k) * T(t);

        residuals[0] = x - T(xi);
        residuals[1] = y - T(yi);
        return true;
    }
    double t, xi, yi;
    double x0, y0;
};

int main(int, char**) {
    cv::VideoCapture cap("../video.mp4");

    const double fps = 60.0;
    std::vector<SamplePoint> samples;
    
    cv::Mat frame;
    int frameIdx = 0;
    const int maxFrames = 180;
    int frameHeight = 0;
    int frameWidth = 0;

    // 检测轨迹
    while (cap.read(frame) && frameIdx < maxFrames) {
        if (frameHeight == 0) {
            frameHeight = frame.rows;
            frameWidth = frame.cols;
        }
        
        cv::Point2d detected = detectBlueProjectile(frame);
        
        if (detected.x >= 0 && detected.y >= 0) {
            double t = frameIdx / fps;
            // 坐标转换：OpenCV像素坐标 → 物理坐标系（原点左下角，Y向上）
            double x_phys = detected.x;
            double y_phys = frameHeight - detected.y; // Y轴翻转
            samples.push_back({t, {x_phys, y_phys}});
        }
        
        frameIdx++;
    }
    
    cap.release();
    
    // 拟合
    const double x0 = samples.front().px.x;
    const double y0 = samples.front().px.y;
    
    ceres::Problem problem;
    double params[4] = { 300.0, 200.0, 500.0, 0.1 }; // vx0, vy0, g, k 初值（物理坐标系）
    
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        auto* cost = new ceres::AutoDiffCostFunction<TrajModelCost, 2, 4>(
            new TrajModelCost(s.t, s.px.x, s.px.y, x0, y0));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(3.0), params);
    }
    
    // 变量边界
    problem.SetParameterLowerBound(params, 2, 100.0);   // g
    problem.SetParameterUpperBound(params, 2, 1000.0);
    problem.SetParameterLowerBound(params, 3, 0.01);    // k
    problem.SetParameterUpperBound(params, 3, 1.0);
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "vx0: " << params[0] << " vy0: " << params[1] << " g: " << params[2] << " k: " << params[3] << std::endl;
    
    // 在原视频尺寸上绘制轨迹图
    cv::Mat trajectoryImg(frameHeight, frameWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 坐标转换函数：物理坐标系 → OpenCV像素坐标
    auto toImgCoords = [&](double x, double y) -> cv::Point {
        int imgX = static_cast<int>(x);
        // 物理坐标系Y向上，OpenCV图像Y向下，需要翻转
        int imgY = frameHeight - static_cast<int>(y);
        return {imgX, imgY};
    };
    
    // 绘制拟合曲线
    if (samples.size() > 0) {
        cv::Point prev = toImgCoords(x0, y0);
        for (double t = 0; t <= samples.back().t + 0.5; t += 0.01) {
            const double expkt = std::exp(-params[3] * t);
            const double x = x0 + (params[0] / params[3]) * (1.0 - expkt);
            const double y = y0 + ((params[1] + params[2] / params[3]) / params[3]) * (1.0 - expkt) - (params[2] / params[3]) * t;
            
            cv::Point curr = toImgCoords(x, y);
            cv::line(trajectoryImg, prev, curr, cv::Scalar(0, 255, 0), 2);
            prev = curr;
        }
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        cv::Point pt = toImgCoords(samples[i].px.x, samples[i].px.y);
        cv::circle(trajectoryImg, pt, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite("../trajectory_result.png", trajectoryImg);
    
    return 0;
}
