package com.github.exercise.service;

import com.github.exercise.data.VideoUpload;

public interface AnalysisService {
    void analyzeVideoAsync(VideoUpload saved);
}
