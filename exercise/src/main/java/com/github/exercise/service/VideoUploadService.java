package com.github.exercise.service;

import com.github.exercise.constants.VideoStatus;
import com.github.exercise.data.User;
import com.github.exercise.data.VideoUpload;
import com.github.exercise.repositories.VideoUploadRepo;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

public class VideoUploadService {
    private final VideoUploadRepo videoUploadRepository;
    private final FileStorageService fileStorageService;
    private final ExerciseAnalysisService exerciseAnalysisService;

    public VideoUploadService(
            VideoUploadRepo videoUploadRepository,
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
        videoUpload.setFileName(file.getOriginalFilename());
        videoUpload.setFileUrl(fileUrl);
        videoUpload.setFileSize(file.getSize());
        videoUpload.setStatus(VideoStatus.PROCESSING);

        VideoUpload saved = videoUploadRepository.save(videoUpload);

        // Trigger async analysis
        exerciseAnalysisService.analyzeVideoAsync(saved);

        return saved;
    }

    public List<VideoUpload> getUserVideos(User user) {
        return videoUploadRepository.findByUserOrderByUploadedAtDesc(user);
    }

    public List<VideoUpload> getPendingVideos() {
        return videoUploadRepository.findByStatus(VideoStatus.PROCESSING);
    }

    @Transactional
    public void updateVideoStatus(Long videoId, VideoStatus status, String errorMessage) {
        VideoUpload video = videoUploadRepository.findById(videoId)
                .orElseThrow(() -> new IllegalArgumentException("Video not found"));

        video.setStatus(status);
        if (errorMessage != null) {
            video.setErrorMessage(errorMessage);
        }

        videoUploadRepository.save(video);
    }
}
