/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
_4A2AAudioProcessor::_4A2AAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(
          BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
              .withInput("Input", juce::AudioChannelSet::mono(), true)
#endif
              .withOutput("Output", juce::AudioChannelSet::mono(), true)
#endif
              )
#endif
      ,
      paramState(
          *this, nullptr, "state",
          {std::make_unique<juce::AudioParameterFloat>(
               juce::ParameterID{"th", 1}, "threshold", -60.0f, 0.0f, -30.0f),
           std::make_unique<juce::AudioParameterFloat>(
               juce::ParameterID{"ratio", 1}, "ratio", 1.0f, 20.0f, 1.0f),
           std::make_unique<juce::AudioParameterFloat>(
               juce::ParameterID{"at", 1}, "attack", 0.1f, 100.0f, 1.0f),
           std::make_unique<juce::AudioParameterFloat>(
               juce::ParameterID{"rt", 1}, "release", 100.0f, 1000.0f, 200.f),
           std::make_unique<juce::AudioParameterFloat>(
               juce::ParameterID{"makeUp", 1}, "makeUp", -12.0f, 12.0f,
               0.0f)})
{
}

_4A2AAudioProcessor::~_4A2AAudioProcessor() {}

//==============================================================================
const juce::String _4A2AAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool _4A2AAudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool _4A2AAudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool _4A2AAudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double _4A2AAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int _4A2AAudioProcessor::getNumPrograms()
{
    return 1; // NB: some hosts don't cope very well if you tell them there are
              // 0 programs, so this should be at least 1, even if you're not
              // really implementing programs.
}

int _4A2AAudioProcessor::getCurrentProgram() { return 0; }

void _4A2AAudioProcessor::setCurrentProgram(int index) {}

const juce::String _4A2AAudioProcessor::getProgramName(int index) { return {}; }

void _4A2AAudioProcessor::changeProgramName(int index,
                                            const juce::String &newName) {}

//==============================================================================
void _4A2AAudioProcessor::prepareToPlay(double sampleRate,
                                        int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    gPrev = 1;
    this->sampleRate = sampleRate;
    gBuffer.resize(samplesPerBlock);
}

void _4A2AAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool _4A2AAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;

        // This checks if the input layout matches the output layout
#if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
#endif
}
#endif

void _4A2AAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                       juce::MidiBuffer &midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    auto totalNumSamples = buffer.getNumSamples();

    auto thValue = paramState.getParameter("th")->convertFrom0to1(
        paramState.getParameter("th")->getValue());
    auto ratioValue = paramState.getParameter("ratio")->convertFrom0to1(
        paramState.getParameter("ratio")->getValue());
    auto atValue =
        paramState.getParameter("at")->convertFrom0to1(paramState.getParameter("at")->getValue());
    auto rtValue =
        paramState.getParameter("rt")->convertFrom0to1(paramState.getParameter("rt")->getValue());
    auto makeUpValue = paramState.getParameter("makeUp")->convertFrom0to1(
        paramState.getParameter("makeUp")->getValue());

    auto thGain = juce::Decibels::decibelsToGain(thValue);
    auto makeUpGain = juce::Decibels::decibelsToGain(makeUpValue);
    auto at = m2c(atValue);
    auto rt = m2c(rtValue);
    auto compSlope = 1.0f - 1.0f / ratioValue;

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, totalNumSamples);

    auto *channelData = buffer.getWritePointer(0);

    std::inclusive_scan(
        channelData, channelData + totalNumSamples, gBuffer.begin(),
        [&thGain, &compSlope, &at, &rt](auto gPrev, auto x)
        {
            auto xAbs = abs(x);
            auto g = std::fmin(1.0f, std::pow(xAbs / thGain, -compSlope));
            auto coef = g < gPrev ? at : rt;
            return g * coef + gPrev * (1 - coef);
        },
        gPrev);
    gPrev = gBuffer.back();
    std::transform(gBuffer.cbegin(), gBuffer.cend(), channelData, channelData,
                   std::multiplies<>{});
    buffer.applyGain(makeUpGain);
}

//==============================================================================
bool _4A2AAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor *_4A2AAudioProcessor::createEditor()
{
    return new _4A2AAudioProcessorEditor(*this, paramState);
}

//==============================================================================
void _4A2AAudioProcessor::getStateInformation(juce::MemoryBlock &destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void _4A2AAudioProcessor::setStateInformation(const void *data,
                                              int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory
    // block, whose contents will have been created by the getStateInformation()
    // call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter()
{
    return new _4A2AAudioProcessor();
}
