package com.github.data;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.springframework.data.annotation.CreatedDate;

import java.time.LocalDateTime;

@Entity
@Table(name = "exercise_analysis")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExcerciseAnalysis {

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
    private Excercise exerciseType;

    @Column(nullable = true)
    private Integer repsPerformed;

    private Double confidence;

    @Column(columnDefinition = "TEXT")
    private String rawResponse;

    @CreationTimestamp
    @Column(nullable = true, updatable = false)
    private LocalDateTime analysisTimestamp;

}
