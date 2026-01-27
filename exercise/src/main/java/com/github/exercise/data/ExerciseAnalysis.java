package com.github.exercise.data;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "exercise_analysis")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExerciseAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "video_upload_id", nullable = true)
    private VideoUpload videoUpload;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = true)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "exercise_id", nullable = true)
    private Exercise exerciseType;

    @Column(nullable = true)
    private Integer repsPerformed;

    private Double confidence;

    @Column(columnDefinition = "TEXT")
    private String rawResponse;

    @CreationTimestamp
    @Column(nullable = true, updatable = false)
    private LocalDateTime analysisTimestamp;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public VideoUpload getVideoUpload() {
        return videoUpload;
    }

    public void setVideoUpload(VideoUpload videoUpload) {
        this.videoUpload = videoUpload;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Exercise getExerciseType() {
        return exerciseType;
    }

    public void setExerciseType(Exercise exerciseType) {
        this.exerciseType = exerciseType;
    }

    public Integer getRepsPerformed() {
        return repsPerformed;
    }

    public void setRepsPerformed(Integer repsPerformed) {
        this.repsPerformed = repsPerformed;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getRawResponse() {
        return rawResponse;
    }

    public void setRawResponse(String rawResponse) {
        this.rawResponse = rawResponse;
    }

    public LocalDateTime getAnalysisTimestamp() {
        return analysisTimestamp;
    }

    public void setAnalysisTimestamp(LocalDateTime analysisTimestamp) {
        this.analysisTimestamp = analysisTimestamp;
    }
}
