/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <algorithm>
#include <vector>

//==============================================================================
/**
 */
class _4A2AAudioProcessor : public juce::AudioProcessor
{
public:
  //==============================================================================
  _4A2AAudioProcessor();
  ~_4A2AAudioProcessor() override;

  //==============================================================================
  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
  bool isBusesLayoutSupported(const BusesLayout &layouts) const override;
#endif

  void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

  //==============================================================================
  juce::AudioProcessorEditor *createEditor() override;
  bool hasEditor() const override;

  //==============================================================================
  const juce::String getName() const override;

  bool acceptsMidi() const override;
  bool producesMidi() const override;
  bool isMidiEffect() const override;
  double getTailLengthSeconds() const override;

  //==============================================================================
  int getNumPrograms() override;
  int getCurrentProgram() override;
  void setCurrentProgram(int index) override;
  const juce::String getProgramName(int index) override;
  void changeProgramName(int index, const juce::String &newName) override;

  //==============================================================================
  void getStateInformation(juce::MemoryBlock &destData) override;
  void setStateInformation(const void *data, int sizeInBytes) override;

private:
  juce::AudioProcessorValueTreeState paramState;
  float gPrev, sampleRate;
  std::vector<float> gBuffer;

  std::atomic<float> *thParam = nullptr;
  std::atomic<float> *ratioParam = nullptr;
  std::atomic<float> *atParam = nullptr;
  std::atomic<float> *rtParam = nullptr;
  std::atomic<float> *makeUpParam = nullptr;

  float m2c(float m) { return 1.0f - std::exp(-2200.0f / m / sampleRate); }
  //==============================================================================
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(_4A2AAudioProcessor)
};
