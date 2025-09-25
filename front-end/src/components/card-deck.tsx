import { useState, ReactNode } from 'react';
import { motion } from 'motion/react';

interface CardDeckProps {
  children: ReactNode[];
}

interface DeckCardProps {
  children: ReactNode;
  index: number;
  isActive: boolean;
  isHovered: boolean;
  onClick: () => void;
  onHover: (hovered: boolean) => void;
  totalCards: number;
}

function DeckCard({ children, index, isActive, isHovered, onClick, onHover, totalCards }: DeckCardProps) {
  const zIndex = isActive ? 30 : totalCards - index + 10;
  
  // Position cards so they peek from the right - more spread out
  const xOffset = isActive ? 0 : index * 80; // Cards peek 80px from right (was 40px)
  const yOffset = isActive ? 0 : index * 4; // Slight vertical offset
  const rotation = isActive ? 0 : index * 1.5; // Slightly more rotation
  
  // Scale effect - make the differences more pronounced
  const scale = isActive ? 1 : isHovered ? 1.08 : 0.85;

  return (
    <motion.div
      className="absolute top-0 left-0 w-full h-full cursor-pointer"
      style={{ zIndex }}
      initial={false}
      animate={{
        x: xOffset,
        y: yOffset,
        rotate: rotation,
        scale: scale,
      }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 30,
      }}
      onClick={onClick}
      onHoverStart={() => onHover(true)}
      onHoverEnd={() => onHover(false)}
    >
      <div className={`
        w-full h-full rounded-lg overflow-hidden
        ${isActive ? 'shadow-2xl shadow-red-500/20' : 'shadow-lg shadow-black/50'}
        ${isHovered && !isActive ? 'shadow-xl shadow-red-500/15' : ''}
        transition-shadow duration-300
      `}>
        {children}
      </div>
    </motion.div>
  );
}

export function CardDeck({ children }: CardDeckProps) {
  const [activeCard, setActiveCard] = useState(0);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  return (
    <div className="relative w-full h-[450px] max-w-3xl mx-auto">
      {/* Card indicators */}
      <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 flex gap-2 z-40">
        {children.map((_, index) => (
          <button
            key={index}
            onClick={() => setActiveCard(index)}
            className={`w-2 h-2 rounded-full transition-colors ${
              activeCard === index
                ? 'bg-red-500'
                : 'bg-white/30 hover:bg-white/50'
            }`}
          />
        ))}
      </div>

      {/* Navigation arrows */}
      <button
        onClick={() => setActiveCard((prev) => (prev > 0 ? prev - 1 : children.length - 1))}
        className="absolute left-2 top-1/2 transform -translate-y-1/2 z-40 bg-gray-900 hover:bg-gray-800 text-white p-2 rounded-full transition-colors border border-white/10"
      >
        ←
      </button>
      <button
        onClick={() => setActiveCard((prev) => (prev < children.length - 1 ? prev + 1 : 0))}
        className="absolute -right-6 top-1/2 transform -translate-y-1/2 z-40 bg-gray-900 hover:bg-gray-800 text-white p-2 rounded-full transition-colors border border-white/10"
      >
        →
      </button>

      {/* Cards Container */}
      <div className="relative w-full h-full overflow-visible pr-32">
        {/* Render cards in reverse order so the first card appears on top */}
        {children.map((child, index) => (
          <DeckCard
            key={index}
            index={index}
            isActive={activeCard === index}
            isHovered={hoveredCard === index}
            onClick={() => setActiveCard(index)}
            onHover={(hovered) => setHoveredCard(hovered ? index : null)}
            totalCards={children.length}
          >
            {child}
          </DeckCard>
        )).reverse()}
      </div>

      {/* Card titles at bottom */}
      <div className="absolute -bottom-16 left-1/2 transform -translate-x-1/2 z-40">
        <div className="text-center">
          <div className="flex gap-6 text-sm">
            {['Record Workout', 'Upload Videos', 'Fitness Journey'].map((title, index) => (
              <button
                key={index}
                onClick={() => setActiveCard(index)}
                className={`transition-colors px-3 py-1 rounded ${
                  activeCard === index 
                    ? 'text-red-400 bg-red-400/10 border border-red-400/20' 
                    : 'text-white/60 hover:text-white/80 hover:bg-white/5'
                }`}
              >
                {title}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}