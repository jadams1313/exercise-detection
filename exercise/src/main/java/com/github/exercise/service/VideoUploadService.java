package com.github.exercise.service;

import com.github.exercise.constants.VideoStatus;
import com.github.exercise.data.User;
import com.github.exercise.data.VideoUpload;
import com.github.exercise.repositories.VideoUploadRepository;
import com.github.exercise.util.S3Util;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import com.github.exercise.service.FileStorageService;

import java.io.IOException;
import java.util.List;

public class VideoUploadService {
    private final VideoUploadRepository videoUploadRepository;
    private final FileStorageService fileStorageService;
    private final ExerciseAnalysisService exerciseAnalysisService;

    public VideoUploadService(
            VideoUploadRepository videoUploadRepository,
            FileStorageService fileStorageService,
            ExerciseAnalysisService exerciseAnalysisService) {
        this.videoUploadRepository = videoUploadRepository;
        this.fileStorageService = fileStorageService;
        this.exerciseAnalysisService = exerciseAnalysisService;
    }

    @Transactional
    public VideoUpload uploadVideo(User user, MultipartFile file) throws IOException {
        // Validate file
        if (file.isEmpty()) {
            throw new IllegalArgumentException("File is empty");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("video/")) {
            throw new IllegalArgumentException("File must be a video");
        }

        // Store file and get URL
        String fileUrl = fileStorageService.storeFile(file);

        // Create video upload record
        VideoUpload videoUpload = new VideoUpload();
        videoUpload.setUser(user);
        videoUpload.setFilename(file.getOriginalFilename());
        videoUpload.setFileUrl(fileUrl);
        videoUpload.setFileSizeBytes(file.getSize());
        videoUpload.setStatus(VideoStatus.PROCESSING);

        VideoUpload saved = videoUploadRepository.save(videoUpload);

        // Trigger async analysis
        exerciseAnalysisService.analyzeVideoAsync(saved);

        return saved;
    }

    public List<VideoUpload> getAllUserVideos(User user) {
        return videoUploadRepository.getAllUserVideos(user);
    }

    public List<VideoUpload> getPendingVideos() {
        return videoUploadRepository.findByStatus(VideoStatus.PROCESSING);
    }

    @Transactional
    public void updateVideoStatus(Long videoId, VideoStatus status, String errorMessage) {
        VideoUpload video = videoUploadRepository.findById(videoId)
                .orElseThrow(() -> new IllegalArgumentException("Video not found"));

        video.setStatus(status);

        videoUploadRepository.save(video);
    }
    // Add these methods to VideoUploadService class

    public VideoUpload getVideoById(Long videoId, Long userId) {
        VideoUpload videoUpload = videoUploadRepository.findById(videoId)
                .orElseThrow(() -> new RuntimeException("Video not found with id: " + videoId));

        // Ensure the video belongs to the requesting user
        if (!videoUpload.getUser().getId().equals(userId)) {
            throw new RuntimeException("Unauthorized access to video");
        }

        return videoUpload;
    }

    public void deleteVideo(Long videoId, Long userId) {
        VideoUpload videoUpload = getVideoById(videoId, userId);

        try {
            // Delete from S3
            fileStorageService.deleteFile(videoUpload.getFilename());

            // Delete from database
            videoUploadRepository.delete(videoUpload);

        } catch (Exception e) {
            throw new RuntimeException("Failed to delete video", e);
        }
    }
}
