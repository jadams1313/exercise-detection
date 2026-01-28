package com.github.exercise.controllers;

import com.github.exercise.data.VideoUpload;
import com.github.exercise.dto.AnalysisResponse;
import com.github.exercise.data.ExerciseAnalysis;
import com.github.exercise.service.ExerciseAnalysisService;
import com.github.exercise.service.VideoUploadService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/analysis")
@RequiredArgsConstructor
public class ExerciseAnalysisController {

    private final ExerciseAnalysisService analysisService;
    private final VideoUploadService videoUploadService;

    @GetMapping("/video/{videoId}")
    public ResponseEntity<AnalysisResponse> getAnalysisByVideoId(
            @PathVariable Long videoId,
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());

        // Verify user owns the video (throws exception if not)
        videoUploadService.getVideoById(videoId, userId);

        // Get the analysis for this video
        ExerciseAnalysis analysis = analysisService.getAnalysisByVideoId(videoId);
        AnalysisResponse response = mapToResponse(analysis);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/{analysisId}")
    public ResponseEntity<AnalysisResponse> getAnalysisById(
            @PathVariable Long analysisId,
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());
        ExerciseAnalysis analysis = analysisService.getAnalysisById(analysisId);

        // Verify user owns the video associated with this analysis
        videoUploadService.getVideoById(analysis.getVideoUpload().getId(), userId);

        AnalysisResponse response = mapToResponse(analysis);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/user")
    public ResponseEntity<List<AnalysisResponse>> getUserAnalyses(
            Authentication authentication) {

        Long userId = Long.parseLong(authentication.getName());

        // Get all analyses for this user
        List<ExerciseAnalysis> analyses = analysisService.getAnalysesByUserId(userId);

        List<AnalysisResponse> responses = analyses.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());

        return ResponseEntity.ok(responses);
    }

    private AnalysisResponse mapToResponse(ExerciseAnalysis analysis) {
        AnalysisResponse response = new AnalysisResponse();
        response.setId(analysis.getId());
        response.setVideoId(analysis.getVideoUpload().getId());
        response.setExercise(analysis.getExerciseType());
        response.setRepCount(analysis.getRepsPerformed());
        response.setConfidence(analysis.getConfidence());
        response.setCreatedAt(analysis.getAnalysisTimestamp());
        return response;
    }
}