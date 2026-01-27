package com.github.exercise.service;

import com.github.exercise.client.VideoAnalysisClient;
import com.github.exercise.data.Exercise;
import com.github.exercise.data.ExerciseAnalysis;
import  com.github.exercise.data.VideoUpload;
import com.github.exercise.data.User;
import com.github.exercise.dto.VideoAnalysisResponse;
import  com.github.exercise.repositories.ExerciseAnalysisRepository;
import com.github.exercise.repositories.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

import static reactor.netty.http.HttpConnectionLiveness.log;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExerciseAnalysisService implements AnalysisService {
    private final ExerciseAnalysisRepository analysisRepository;
    private final VideoAnalysisClient videoAnalysisClient;
    private final FileStorageService fileStorageService;
    private final UserRepository userRepository;

    @Async
    @Transactional
    public void analyzeVideoAsync(VideoUpload videoUpload) {
        log.info("Starting async analysis for video: {}", videoUpload.getId());

        ExerciseAnalysis analysis = new ExerciseAnalysis();
        analysis.setVideoUpload(videoUpload);
        analysis.setAnalysisTimestamp(LocalDateTime.now());
        analysis = analysisRepository.save(analysis);

        try {
            // Get video from S3 and send to ML model
            VideoAnalysisResponse response = videoAnalysisClient.analyzeVideo(videoUpload.getFilename());
            final Exercise exerciseClassificationResponse = new Exercise();
            exerciseClassificationResponse.setType(response.getExerciseType());
            analysis.setExerciseType(exerciseClassificationResponse);
            analysis.setRepsPerformed(response.getRepCount());
            analysis.setConfidence(response.getConfidence());
//          analysis.setStatus(AnalysisStatus.COMPLETED); do we need this?
            analysis.setAnalysisTimestamp(LocalDateTime.now());

            log.info("Analysis completed for video: {} - Type: {}, Reps: {}",
                    videoUpload.getId(), response.getExerciseType(), response.getRepCount());

        } catch (Exception e) {
            log.error("Analysis failed for video: {}", videoUpload.getId(), e);
    //        analysis.setStatus(AnalysisStatus.FAILED);
    //        analysis.setErrorMessage(e.getMessage());
        }

        analysisRepository.save(analysis);
    }

    @Transactional(readOnly = true)
    public ExerciseAnalysis getAnalysisByVideoId(Long videoId) {
        return analysisRepository.findByVideoUploadId(videoId)
                .orElseThrow(() -> new RuntimeException("Analysis not found for video: " + videoId));
    }

    @Transactional(readOnly = true)
    public List<ExerciseAnalysis> getAnalysesByUserId(Long userId) {
        return analysisRepository.findByVideoUploadUserId(userId);
    }

    @Transactional(readOnly = true)
    public ExerciseAnalysis getAnalysisById(Long analysisId) {
        return analysisRepository.findById(analysisId)
                .orElseThrow(() -> new RuntimeException("Analysis not found: " + analysisId));
    }

    @Transactional(readOnly = true)
    public List<ExerciseAnalysis> getAnalysesByUsername(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));
        return analysisRepository.findByVideoUploadUserId(user.getId());
    }
}