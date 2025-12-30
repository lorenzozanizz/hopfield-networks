#pragma once
#ifndef PREPROCESSOR_AID_HPP
#define PREPROCESSOR_AID_HPP

// Small headedr only library of utils for development ease and giving
// the user and developer information during the compilation step.
// see for ex. 
// https://stackoverflow.com/questions/5966594/how-can-i-use-pragma-message-so-that-the-message-points-to-the-filelineno

#define Stringize( L )     #L 
#define MakeString( M, L ) M(L)

#define Reminder __FILE__ "(" MakeString(Stringize, __LINE__) ") : Reminder: "
#define Todo __FILE__ "(" MakeString(Stringize, __LINE__) ") : Todo: "
#define Warning __FILE__ "(" MakeString(Stringize, __LINE__) ") : Warning: "

#endif